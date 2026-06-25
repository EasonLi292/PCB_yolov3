#!/usr/bin/env python3
"""
Evaluate a trained YOLOv3 OpenVINO IR on a local test split: per-class AP@0.5, mAP,
and a montage of sample predictions vs ground truth.

Run this after training in Colab once you've downloaded yolov3.xml + yolov3.bin.

Example:
  python scripts/analyze_openvino.py \
      --ir runs/unified_pku_yolo_gray640/openvino/yolov3.xml \
      --data datasets/unified_pku_yolo_gray640 --split test \
      --classes runs/unified_pku_yolo_gray640/classes.txt
"""
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import openvino as ov

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_openvino import preprocess, postprocess  # reuse the exact inference path


def load_gt(label_path: Path, w, h):
    boxes = []
    if label_path.exists():
        for ln in label_path.read_text().splitlines():
            p = ln.split()
            if len(p) < 5:
                continue
            c, cx, cy, bw, bh = int(float(p[0])), *map(float, p[1:5])
            boxes.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def voc_ap(rec, prec):
    """VOC-style AP: area under the monotonic precision-recall curve."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", required=True, help="path to yolov3.xml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--classes", required=True)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--score", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=12, help="how many vis images to save")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    classes = Path(args.classes).read_text().split()
    img_dir = Path(args.data) / args.split / "images"
    lbl_dir = Path(args.data) / args.split / "labels"
    out_dir = Path(args.out) if args.out else Path(args.ir).parent / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    compiled = ov.Core().compile_model(args.ir, "CPU")
    # Auto-detect IR type: 3 outputs = FPGA raw conv heads (decode on host);
    # 1 output = the decode-baked IR from export_openvino.py.
    fpga_raw = len(compiled.outputs) >= 3
    if fpga_raw:
        from yolo_postprocess import decode_and_nms
    print(f"IR outputs: {len(compiled.outputs)}  ->",
          "FPGA raw heads (host decode)" if fpga_raw else "decoded IR")

    # per-class: list of (score, is_tp); and total GT count
    preds = defaultdict(list)
    n_gt = defaultdict(int)
    imgs = sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    print(f"Evaluating {len(imgs)} {args.split} images on {len(classes)} classes...")

    sample_paths = set(imgs[:: max(1, len(imgs) // args.samples)][:args.samples]) if args.samples else set()
    for k, img_path in enumerate(imgs):
        inp, vis, (h0, w0) = preprocess(img_path, args.size)
        result = compiled(inp)
        if fpga_raw:
            raw = [result[o] for o in compiled.outputs]
            dets = decode_and_nms(raw, w0, h0, args.score, args.iou)
        else:
            pred = result[compiled.output(0)][0]
            dets = postprocess(pred, w0, h0, args.score, args.iou)   # (x1,y1,x2,y2,score,cls)
        gts = load_gt(lbl_dir / f"{img_path.stem}.txt", w0, h0)
        for c, *_ in gts:
            n_gt[c] += 1

        matched = set()
        for x1, y1, x2, y2, s, c in sorted(dets, key=lambda d: -d[4]):
            best_iou, best_j = 0.0, -1
            for j, (gc, gx1, gy1, gx2, gy2) in enumerate(gts):
                if gc != c or j in matched:
                    continue
                v = iou((x1, y1, x2, y2), (gx1, gy1, gx2, gy2))
                if v > best_iou:
                    best_iou, best_j = v, j
            is_tp = best_iou >= args.iou and best_j >= 0
            if is_tp:
                matched.add(best_j)
            preds[c].append((s, is_tp))

        if img_path in sample_paths:
            for gc, gx1, gy1, gx2, gy2 in gts:  # GT = green
                cv2.rectangle(vis, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0, 200, 0), 1)
            for x1, y1, x2, y2, s, c in dets:    # pred = red
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(vis, f"{classes[c]} {s:.2f}", (x1, max(0, y1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            # vis is a 3-channel gray canvas; colors were drawn in BGR -> save directly
            cv2.imwrite(str(out_dir / f"sample_{img_path.stem}.jpg"), vis)

    # per-class AP
    print(f"\n{'class':18s} {'GT':>6s} {'preds':>6s} {'AP@0.5':>8s}")
    aps = []
    for ci, cname in enumerate(classes):
        dets = sorted(preds.get(ci, []), key=lambda d: -d[0])
        tp = np.array([d[1] for d in dets], dtype=np.float32)
        fp = 1 - tp
        if n_gt[ci] == 0:
            print(f"{cname:18s} {0:6d} {len(dets):6d} {'  n/a':>8s}")
            continue
        tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
        rec = tp_c / (n_gt[ci] + 1e-9)
        prec = tp_c / (tp_c + fp_c + 1e-9)
        a = voc_ap(rec, prec) if len(dets) else 0.0
        aps.append(a)
        print(f"{cname:18s} {n_gt[ci]:6d} {len(dets):6d} {a:8.3f}")
    mAP = float(np.mean(aps)) if aps else 0.0
    print(f"\nmAP@0.5 = {mAP:.3f}   (sample visualizations -> {out_dir})")


if __name__ == "__main__":
    main()
