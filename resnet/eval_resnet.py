#!/usr/bin/env python3
"""
Evaluate the ResNet-50 good/bad classifier on the held-out test split.

Reports the metrics that matter for an imbalanced good/bad screen:
  * confusion matrix at a chosen threshold
  * precision / recall / F1 for the DEFECTIVE (positive) class
  * accuracy, ROC-AUC, PR-AUC (threshold-independent)
  * a threshold sweep, so you can pick the operating point (e.g. high recall to
    avoid shipping bad boards, accepting more false alarms)

Usage:
  python resnet/eval_resnet.py --weights runs_resnet/pcb_goodbad/best.weights.h5 \
         --data datasets/pcb_goodbad --threshold 0.5
"""
import argparse, os, sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import tensorflow as tf
from resnet50_tf import build_resnet50
from data import make_dataset


_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))   # NumPy 2 renamed trapz


def auc_trapz(xs, ys):
    order = np.argsort(xs)
    return float(_trapz(np.array(ys)[order], np.array(xs)[order]))


def curves(y, p):
    """Return (roc_auc, pr_auc). ROC via trapezoid over thresholds; PR-AUC as the
    standard step-wise Average Precision (rank scores desc, sum prec*Δrecall) — the
    threshold-trapezoid form mis-integrates the near-vertical PR curve of a strong
    classifier, so we use AP instead."""
    thr = np.unique(np.concatenate([[0.0], p, [1.0]]))
    tpr, fpr = [], []
    P, N = (y == 1).sum(), (y == 0).sum()
    for t in thr:
        pred = p >= t
        tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
        tpr.append(tp / P if P else 0.0)
        fpr.append(fp / N if N else 0.0)
    order = np.argsort(-p)
    ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    rec = tp / P if P else np.zeros_like(tp, float)
    prec = tp / np.maximum(tp + fp, 1)
    ap = float(prec[0] * rec[0] + np.sum((rec[1:] - rec[:-1]) * prec[1:]))
    return auc_trapz(fpr, tpr), ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    model = build_resnet50(size=args.size, freeze_backbone=True)
    model.load_weights(args.weights)

    ds, n, counts = make_dataset(Path(args.data) / args.split, args.size, args.batch)
    print(f"{args.split}: {n} images  {counts}")

    y, p = [], []
    for xb, yb in ds:
        p.append(model.predict(xb, verbose=0).ravel())
        y.append(yb.numpy().ravel())
    y = np.concatenate(y).astype(int)
    p = np.concatenate(p)

    t = args.threshold
    pred = (p >= t).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    acc = (tp + tn) / max(len(y), 1)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    roc_auc, pr_auc = curves(y, p)

    print(f"\n== Confusion matrix @ threshold {t:.2f}  (positive = defective) ==")
    print(f"               pred good   pred bad")
    print(f"  actual good   {tn:7d}    {fp:7d}")
    print(f"  actual bad    {fn:7d}    {tp:7d}")
    print(f"\naccuracy            : {acc:.3f}")
    print(f"precision (defect)  : {prec:.3f}")
    print(f"recall    (defect)  : {rec:.3f}   <- fraction of bad boards caught")
    print(f"F1        (defect)  : {f1:.3f}")
    print(f"ROC-AUC             : {roc_auc:.3f}")
    print(f"PR-AUC  (defect)    : {pr_auc:.3f}")

    print("\n== threshold sweep ==")
    print("  thr   acc   prec   recall")
    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        pr = (p >= t).astype(int)
        tp = np.sum((pr == 1) & (y == 1)); fp = np.sum((pr == 1) & (y == 0))
        fn = np.sum((pr == 0) & (y == 1)); tn = np.sum((pr == 0) & (y == 0))
        a = (tp + tn) / max(len(y), 1)
        pc = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        print(f"  {t:.1f}  {a:.3f}  {pc:.3f}  {rc:.3f}")


if __name__ == "__main__":
    main()
