#!/usr/bin/env python3
"""
Compute YOLOv3 anchors by k-means (IoU distance) over a dataset's train boxes, and write
them to scripts/anchors.json so training + host decode both pick them up.

Stock COCO anchors are far larger than tiny PCB defects; fitting anchors to the actual box
sizes improves anchor->box matching and small-object recall. Anchors live only in training
target-assignment and host-side decode, so this has ZERO effect on the FPGA conv graph.

    python scripts/compute_anchors.py --data datasets/unified_pku_yolo_gray640
"""
import argparse
import json
from pathlib import Path
import numpy as np


def iou_wh(boxes, clusters):
    """IoU between each (w,h) box and each (w,h) cluster, aligned at a common corner."""
    mw = np.minimum(boxes[:, None, 0], clusters[None, :, 0])
    mh = np.minimum(boxes[:, None, 1], clusters[None, :, 1])
    inter = mw * mh
    ab = (boxes[:, 0] * boxes[:, 1])[:, None]
    ac = (clusters[:, 0] * clusters[:, 1])[None, :]
    return inter / (ab + ac - inter + 1e-12)


def kmeans(boxes, k, iters=500, seed=0):
    rng = np.random.default_rng(seed)
    clusters = boxes[rng.choice(len(boxes), k, replace=False)].copy()
    last = None
    for _ in range(iters):
        assign = (1 - iou_wh(boxes, clusters)).argmin(1)
        if last is not None and np.array_equal(assign, last):
            break
        for i in range(k):
            if (assign == i).any():
                clusters[i] = boxes[assign == i].mean(0)
        last = assign
    return clusters


def load_boxes(data_dir, split):
    wh = []
    for lf in (Path(data_dir) / split / "labels").glob("*.txt"):
        for ln in lf.read_text().splitlines():
            p = ln.split()
            if len(p) >= 5:
                wh.append((float(p[3]), float(p[4])))
    return np.array(wh, np.float32)


def avg_iou(boxes, clusters):
    return float(iou_wh(boxes, clusters).max(1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--k", type=int, default=9)
    ap.add_argument("--out", default=str(Path(__file__).with_name("anchors.json")))
    args = ap.parse_args()

    boxes = load_boxes(args.data, args.split)
    print(f"{len(boxes)} boxes from {args.data}/{args.split}")

    clusters = kmeans(boxes, args.k)
    clusters = clusters[np.argsort(clusters[:, 0] * clusters[:, 1])]   # ascending area

    stock = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                      (59, 119), (116, 90), (156, 198), (373, 326)], np.float32) / 416.0
    print(f"mean IoU (boxes vs best anchor)  stock: {avg_iou(boxes, stock):.3f}   "
          f"k-means: {avg_iou(boxes, clusters):.3f}")

    masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    out = {"anchors": [[round(float(w), 5), round(float(h), 5)] for w, h in clusters],
           "masks": masks, "source": f"{args.data}/{args.split}", "k": args.k}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    for w, h in clusters:
        print(f"  {w:.4f} x {h:.4f}   (area {w*h*100:.3f}% of image)")


if __name__ == "__main__":
    main()
