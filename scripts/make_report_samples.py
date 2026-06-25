#!/usr/bin/env python3
"""
Build a sample montage for the model report: run the trained IR on test images and draw
ground-truth boxes (GREEN) vs model predictions (RED) on a grid.

    python scripts/make_report_samples.py \
        --ir runs/unified_pku_yolo_gray640/openvino_fpga/yolov3_fpga_fp32.xml \
        --data datasets/unified_pku_yolo_gray640 --split test \
        --classes runs/unified_pku_yolo_gray640/openvino_fpga/classes.txt \
        --out docs/eval_samples.jpg --n 15
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
import openvino as ov

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_openvino import preprocess
from analyze_openvino import load_gt
from yolo_postprocess import decode_and_nms


def label_strip(img, text, h=22):
    strip = np.full((h, img.shape[1], 3), 40, np.uint8)
    cv2.putText(strip, text, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    return np.vstack([strip, img])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--classes", required=True)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--score", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--cell", type=int, default=360)
    ap.add_argument("--out", default="docs/eval_samples.jpg")
    args = ap.parse_args()

    classes = Path(args.classes).read_text().split()
    img_dir = Path(args.data) / args.split / "images"
    lbl_dir = Path(args.data) / args.split / "labels"
    compiled = ov.Core().compile_model(args.ir, "CPU")
    raw_ir = len(compiled.outputs) >= 3

    imgs = sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    # spread across the set so samples aren't all the same board
    imgs = imgs[:: max(1, len(imgs) // (args.n * 6))]

    cells = []
    for img_path in imgs:
        gts = load_gt(lbl_dir / f"{img_path.stem}.txt", 1, 1)  # normalized
        if not gts:
            continue
        inp, vis, (h0, w0) = preprocess(img_path, args.size)
        if raw_ir:
            dets = decode_and_nms([compiled(inp)[o] for o in compiled.outputs],
                                  w0, h0, args.score, args.iou)
        else:
            dets = []
        # prefer images where the model actually fired something
        if not dets:
            continue
        for c, gx1, gy1, gx2, gy2 in gts:           # GREEN = ground truth
            cv2.rectangle(vis, (int(gx1 * w0), int(gy1 * h0)),
                          (int(gx2 * w0), int(gy2 * h0)), (0, 200, 0), 1)
        for x1, y1, x2, y2, s, c in dets:            # RED = prediction
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis, f"{classes[c]} {s:.2f}", (x1, max(10, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cell = cv2.resize(vis, (args.cell, args.cell))
        cells.append(label_strip(cell, f"{img_path.stem[:34]}  ({len(gts)} GT, {len(dets)} pred)"))
        if len(cells) >= args.n:
            break

    if not cells:
        raise SystemExit("no samples with both GT and predictions found")

    cols = args.cols
    rows = (len(cells) + cols - 1) // cols
    ch, cw = cells[0].shape[:2]
    grid = np.full((rows * ch, cols * cw, 3), 20, np.uint8)
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        grid[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = cell

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid)
    print(f"wrote {len(cells)} samples -> {out}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
