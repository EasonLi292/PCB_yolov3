#!/usr/bin/env python3
"""
Build sample montages for the model report. Each sample is a PAIR of panels:
left = ground truth (GREEN), right = model prediction (RED), both with legible class
labels. The N samples are split into several smaller chunk images (easier to read than
one giant grid).

    python scripts/make_report_samples.py \
        --ir runs/unified_pku_yolo_gray640/openvino_fpga/yolov3_fpga_fp32.xml \
        --data datasets/unified_pku_yolo_gray640 --split test \
        --classes runs/unified_pku_yolo_gray640/openvino_fpga/classes.txt \
        --out docs/eval_samples.jpg --n 50 --chunk 10
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

SOURCES = [("hr_", "HRIPCB"), ("nb_", "norbertelter"), ("rf_", "Roboflow")]


def source_of(stem):
    for pref, name in SOURCES:
        if stem.startswith(pref):
            return name
    return "other"


def draw_label(img, text, x, y, color, scale=0.6):
    """Legible label: filled colored background + white text."""
    th = 1
    (tw, tht), _ = cv2.getTextSize(text, FONT, scale, th + 1)
    y = max(tht + 4, y)
    x = max(0, min(x, img.shape[1] - tw - 4))
    cv2.rectangle(img, (x, y - tht - 5), (x + tw + 4, y + 2), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), FONT, scale, (255, 255, 255), th, cv2.LINE_AA)


def caption(img, text, color, h=30):
    strip = np.full((h, img.shape[1], 3), 35, np.uint8)
    cv2.putText(strip, text, (8, 21), FONT, 0.62, color, 2, cv2.LINE_AA)
    return np.vstack([strip, img])


def name_bar(width, text, h=26):
    bar = np.full((h, width, 3), 20, np.uint8)
    cv2.putText(bar, text, (8, 18), FONT, 0.52, (225, 225, 225), 1, cv2.LINE_AA)
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
    ap.add_argument("--chunk", type=int, default=10, help="pairs per output image")
    ap.add_argument("--pairs-per-row", type=int, default=2)
    ap.add_argument("--cell", type=int, default=400)
    ap.add_argument("--thick", type=int, default=3)
    ap.add_argument("--out", default="docs/eval_samples.jpg")
    args = ap.parse_args()

    classes = Path(args.classes).read_text().split()
    img_dir = Path(args.data) / args.split / "images"
    lbl_dir = Path(args.data) / args.split / "labels"
    compiled = ov.Core().compile_model(args.ir, "CPU")
    raw_ir = len(compiled.outputs) >= 3

    imgs = sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    by_src = {}
    for p in imgs:
        by_src.setdefault(source_of(p.stem), []).append(p)
    queues = {k: deque(v[:: max(1, len(v) // (args.n * 2))]) for k, v in by_src.items()}
    order = [name for _, name in SOURCES if name in queues] + \
            [k for k in queues if k not in {n for _, n in SOURCES}]

    def build_pair(img_path):
        gts = load_gt(lbl_dir / f"{img_path.stem}.txt", 1, 1)
        if not gts:
            return None
        inp, vis, (h0, w0) = preprocess(img_path, args.size)
        dets = decode_and_nms([compiled(inp)[o] for o in compiled.outputs],
                              w0, h0, args.score, args.iou) if raw_ir else []
        if not dets:
            return None
        gt_img, pred_img = vis.copy(), vis.copy()
        for c, gx1, gy1, gx2, gy2 in gts:
            p1, p2 = (int(gx1 * w0), int(gy1 * h0)), (int(gx2 * w0), int(gy2 * h0))
            cv2.rectangle(gt_img, p1, p2, GREEN, args.thick)
            draw_label(gt_img, classes[c], p1[0], p1[1], GREEN)
        for x1, y1, x2, y2, s, c in dets:
            cv2.rectangle(pred_img, (x1, y1), (x2, y2), RED, args.thick)
            draw_label(pred_img, f"{classes[c]} {s:.2f}", x1, y1, RED)
        cell = args.cell
        left = caption(cv2.resize(gt_img, (cell, cell)), f"GROUND TRUTH ({len(gts)})", GREEN)
        right = caption(cv2.resize(pred_img, (cell, cell)), f"PREDICTION ({len(dets)})", RED)
        sep = np.full((left.shape[0], 6, 3), 20, np.uint8)
        pair = np.hstack([left, sep, right])
        return np.vstack([name_bar(pair.shape[1],
                          f"[{source_of(img_path.stem)}]  {img_path.stem[:52]}"), pair])

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

    # split into chunks and write one image per chunk
    out = Path(args.out)
    cols = args.pairs_per_row
    written = []
    for ci in range(0, len(pairs), args.chunk):
        group = pairs[ci:ci + args.chunk]
        rows = (len(group) + cols - 1) // cols
        ph, pw = group[0].shape[:2]
        gap = 8
        grid = np.full((rows * ph + (rows - 1) * gap, cols * pw + (cols - 1) * gap, 3), 10, np.uint8)
        for i, pair in enumerate(group):
            r, c = divmod(i, cols)
            grid[r * (ph + gap):r * (ph + gap) + ph, c * (pw + gap):c * (pw + gap) + pw] = pair
        fp = out.with_name(f"{out.stem}_{ci // args.chunk + 1:02d}{out.suffix}")
        fp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(fp), grid)
        written.append(fp.name)
    # clean up any stale single-file montage from older runs
    if out.exists():
        out.unlink()
    print(f"wrote {len(pairs)} pairs across {len(written)} chunks: {', '.join(written)}")


if __name__ == "__main__":
    main()
