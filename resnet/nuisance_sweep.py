#!/usr/bin/env python3
"""
Nuisance-robustness sweep: how much of the test accuracy survives when boards stop being
pixel-identical?

HRIPCB has exactly ONE photograph per board design (verified: pixel-identical outside the
defect boxes), so the test set has ZERO board-to-board variation. A real imaging rig has
sensor noise, gain/lighting drift, focus softness, and sub-pixel registration error. This
script injects each of those at increasing magnitude and reports accuracy / ROC-AUC, so the
"perfect imaging" number can be turned into a spec for how repeatable the rig must be.

Inference only -- no retraining.

Usage:
  python resnet/nuisance_sweep.py --weights <run>/best.weights.h5 --data datasets/pcb_bin_offset --size 512
"""
import argparse, os, sys, glob, json
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, cv2
import tensorflow as tf
from resnet50_tf import build_resnet50, preprocess_batch

rng = np.random.default_rng(0)


# ---- nuisance transforms (applied to the uint8 RGB patch, before preprocessing) ----
def noise(img, sigma):                      # sensor read noise
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)

def gain(img, pct):                         # lighting / exposure drift (multiplicative)
    return np.clip(img.astype(np.float32) * (1.0 + pct), 0, 255).astype(np.uint8)

def blur(img, sigma):                       # focus softness
    return cv2.GaussianBlur(img, (0, 0), sigma) if sigma > 0 else img

def shift(img, px):                         # sub-pixel registration error
    if px == 0: return img
    M = np.float32([[1, 0, px], [0, 1, px]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)

def rot(img, deg):                          # fixture rotation error
    if deg == 0: return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

SWEEPS = {
    "noise (sigma, gray levels)": (noise, [0, 2, 5, 10, 15, 20]),
    "gain (% exposure drift)":    (gain,  [0.0, 0.05, 0.10, 0.20, 0.30]),
    "blur (gaussian sigma px)":   (blur,  [0, 0.5, 1.0, 1.5, 2.0]),
    "shift (px registration)":    (shift, [0, 1, 2, 4, 8]),
    "rotation (degrees)":         (rot,   [0, 0.5, 1.0, 2.0, 4.0]),
}


def load_split(data, split, size):
    paths, ys = [], []
    for cls, y in (("good", 0), ("bad", 1)):
        for p in sorted(glob.glob(str(Path(data) / split / cls / "*.jpg"))):
            paths.append(p); ys.append(y)
    return paths, np.array(ys)


def roc_auc(y, s):
    o = np.argsort(-s); y = y[o]
    P, N = y.sum(), len(y) - y.sum()
    if P == 0 or N == 0: return float("nan")
    tps = np.cumsum(y); fps = np.cumsum(1 - y)
    return float(np.trapezoid(tps / P, fps / N))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="cap patches (0 = all)")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    model = build_resnet50(size=args.size, freeze_backbone=True)
    model.load_weights(args.weights)
    paths, y = load_split(args.data, args.split, args.size)
    if args.limit:
        idx = rng.permutation(len(paths))[:args.limit]
        paths = [paths[i] for i in idx]; y = y[idx]
    imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
    print(f"{args.data}  split={args.split}  n={len(imgs)}  size={args.size}  weights={args.weights}")

    def score_all(fn, mag):
        out = []
        for i in range(0, len(imgs), args.batch):
            chunk = imgs[i:i + args.batch]
            batch = np.stack([cv2.resize(fn(im, mag), (args.size, args.size),
                                         interpolation=cv2.INTER_AREA) for im in chunk]).astype(np.float32)
            out.append(model(preprocess_batch(batch), training=False).numpy().ravel())
        return np.concatenate(out)

    results = {}
    for name, (fn, mags) in SWEEPS.items():
        print(f"\n== {name} ==")
        print(f"  {'magnitude':>12}  {'accuracy':>9}  {'ROC-AUC':>8}  {'recall':>7}  {'precision':>9}")
        rows = []
        for m in mags:
            s = score_all(fn, m)
            pred = (s >= 0.5).astype(int)
            acc = float((pred == y).mean())
            tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
            fn_ = int(((pred == 0) & (y == 1)).sum())
            rec = tp / max(tp + fn_, 1); pre = tp / max(tp + fp, 1)
            auc = roc_auc(y, s)
            print(f"  {m:>12}  {acc:>9.4f}  {auc:>8.4f}  {rec:>7.3f}  {pre:>9.3f}")
            rows.append(dict(magnitude=m, accuracy=acc, roc_auc=auc, recall=rec, precision=pre))
        results[name] = rows

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"weights": args.weights, "data": args.data, "size": args.size,
             "n": len(imgs), "sweeps": results}, indent=2))
        print("\nwrote", args.json_out)


if __name__ == "__main__":
    main()
