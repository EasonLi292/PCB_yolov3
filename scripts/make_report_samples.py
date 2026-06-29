#!/usr/bin/env python3
"""
Build a sample montage for the model report. Each sample is shown as a PAIR of panels:
the left panel draws the ground-truth boxes (GREEN), the right panel draws the model's
predictions (RED) on the same image — so you compare "correct" vs "guessed" side by side.

    python scripts/make_report_samples.py \
        --ir runs/unified_pku_yolo_gray640/openvino_fpga/yolov3_fpga_fp32.xml \
        --data datasets/unified_pku_yolo_gray640 --split test \
        --classes runs/unified_pku_yolo_gray640/openvino_fpga/classes.txt \
        --out docs/eval_samples.jpg --n 15
"""
import argparse
from collections import deque
from pathlib import Path
import numpy as np
import cv2
import openvino as ov

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_openvino import preprocess
from analyze_openvino import load_gt
from yolo_postprocess import decode_and_nms

GREEN = (0, 200, 0)
RED = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Source of each image, by filename prefix (the unified PKU set merges 3 datasets).
SOURCES = [("hr_", "HRIPCB"), ("nb_", "norbertelter"), ("rf_", "Roboflow")]


def source_of(stem):
    for pref, name in SOURCES:
        if stem.startswith(pref):
            return name
    return "other"


def caption(img, text, color, h=26):
    strip = np.full((h, img.shape[1], 3), 35, np.uint8)
    cv2.putText(strip, text, (8, 18), FONT, 0.55, color, 2)
    return np.vstack([strip, img])


def name_bar(width, text, h=24):
    bar = np.full((h, width, 3), 20, np.uint8)
    cv2.putText(bar, text, (8, 17), FONT, 0.5, (220, 220, 220), 1)
    return bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--classes", required=True)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--score", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--pairs-per-row", type=int, default=5)
    ap.add_argument("--cell", type=int, default=300)
    ap.add_argument("--thick", type=int, default=3, help="box thickness")
    ap.add_argument("--out", default="docs/eval_samples.jpg")
    args = ap.parse_args()

    classes = Path(args.classes).read_text().split()
    img_dir = Path(args.data) / args.split / "images"
    lbl_dir = Path(args.data) / args.split / "labels"
    compiled = ov.Core().compile_model(args.ir, "CPU")
    raw_ir = len(compiled.outputs) >= 3

    imgs = sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})

    # Group by source dataset and spread within each, so the montage balances across
    # all three merged sources instead of skewing to whichever sorts first.
    by_src = {}
    for p in imgs:
        by_src.setdefault(source_of(p.stem), []).append(p)
    queues = {}
    for k, v in by_src.items():
        queues[k] = deque(v[:: max(1, len(v) // (args.n * 2))])
    order = [name for _, name in SOURCES if name in queues] + \
            [k for k in queues if k not in {n for _, n in SOURCES}]

    def build_pair(img_path):
        gts = load_gt(lbl_dir / f"{img_path.stem}.txt", 1, 1)   # normalized coords
        if not gts:
            return None
        inp, vis, (h0, w0) = preprocess(img_path, args.size)
        dets = decode_and_nms([compiled(inp)[o] for o in compiled.outputs],
                              w0, h0, args.score, args.iou) if raw_ir else []
        if not dets:                       # prefer images where the model fired
            return None
        gt_img, pred_img = vis.copy(), vis.copy()
        for c, gx1, gy1, gx2, gy2 in gts:
            cv2.rectangle(gt_img, (int(gx1 * w0), int(gy1 * h0)),
                          (int(gx2 * w0), int(gy2 * h0)), GREEN, args.thick)
        for x1, y1, x2, y2, s, c in dets:
            cv2.rectangle(pred_img, (x1, y1), (x2, y2), RED, args.thick)
            cv2.putText(pred_img, f"{classes[c]} {s:.2f}", (x1, max(14, y1 - 4)),
                        FONT, 0.5, RED, 2)
        cell = args.cell
        left = caption(cv2.resize(gt_img, (cell, cell)), f"GROUND TRUTH ({len(gts)})", GREEN)
        right = caption(cv2.resize(pred_img, (cell, cell)), f"PREDICTION ({len(dets)})", RED)
        sep = np.full((left.shape[0], 6, 3), 20, np.uint8)
        pair = np.hstack([left, sep, right])
        return np.vstack([name_bar(pair.shape[1],
                          f"[{source_of(img_path.stem)}]  {img_path.stem[:48]}"), pair])

    # round-robin across sources until we have n pairs
    pairs = []
    while len(pairs) < args.n and any(queues[k] for k in order):
        progressed = False
        for k in order:
            if not queues[k]:
                continue
            pair = build_pair(queues[k].popleft())
            progressed = True
            if pair is not None:
                pairs.append(pair)
                if len(pairs) >= args.n:
                    break
        if not progressed:
            break

    if not pairs:
        raise SystemExit("no samples with both GT and predictions found")

    cols = args.pairs_per_row
    rows = (len(pairs) + cols - 1) // cols
    ph, pw = pairs[0].shape[:2]
    gap = 8
    grid = np.full((rows * ph + (rows - 1) * gap, cols * pw + (cols - 1) * gap, 3), 10, np.uint8)
    for i, pair in enumerate(pairs):
        r, c = divmod(i, cols)
        y, x = r * (ph + gap), c * (pw + gap)
        grid[y:y + ph, x:x + pw] = pair

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid)
    print(f"wrote {len(pairs)} GT-vs-prediction pairs -> {out}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
