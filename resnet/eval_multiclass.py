#!/usr/bin/env python3
"""
Evaluate the multi-class defect-TYPE ResNet on the held-out test split (Goal 2).

Reports overall top-1 accuracy, the full KxK confusion matrix (which defect types get
mistaken for which), and per-class precision / recall / F1.

Usage:
  python resnet/eval_multiclass.py --weights runs_resnet/pcb_defect_types/best.weights.h5 \
         --data datasets/pcb_defect_types --size 256
"""
import argparse, os, sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import tensorflow as tf
from resnet50_tf import build_resnet50
import data_multiclass as dm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    data_dir = Path(args.data)
    names = dm.class_names(data_dir)
    K = len(names)
    model = build_resnet50(size=args.size, freeze_backbone=True, n_classes=K)
    model.load_weights(args.weights)

    ds, n, counts = dm.make_dataset(data_dir / args.split, names, args.size, args.batch)
    print(f"{args.split}: {n} images  {counts}")

    y, P = [], []
    for xb, yb in ds:
        P.append(model.predict(xb, verbose=0)); y.append(yb.numpy())
    y = np.concatenate(y); P = np.concatenate(P); pred = P.argmax(1)

    cm = np.zeros((K, K), int)
    for t, pr in zip(y, pred):
        cm[t, pr] += 1
    acc = float((pred == y).mean())

    print(f"\noverall top-1 accuracy: {acc:.4f}\n")
    print("confusion matrix (rows = true, cols = predicted):")
    print("            " + " ".join(f"{nm[:7]:>8s}" for nm in names))
    for i, nm in enumerate(names):
        print(f"{nm[:11]:>11s} " + " ".join(f"{cm[i, j]:8d}" for j in range(K)))

    print("\nper-class precision / recall / F1:")
    for i, nm in enumerate(names):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        print(f"  {nm:16s} P {pr:.3f}  R {rc:.3f}  F1 {f1:.3f}  (support={cm[i, :].sum()})")
    macro_f1 = np.mean([
        (lambda tp, fp, fn: 2 * (tp / (tp + fp) if tp + fp else 0) * (tp / (tp + fn) if tp + fn else 0) /
         (((tp / (tp + fp)) if tp + fp else 0) + ((tp / (tp + fn)) if tp + fn else 0) + 1e-9))(
            cm[i, i], cm[:, i].sum() - cm[i, i], cm[i, :].sum() - cm[i, i])
        for i in range(K)])
    print(f"\nmacro-F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
