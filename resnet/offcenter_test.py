#!/usr/bin/env python3
"""
Goal 4: does the defect's distance from the patch center affect detection?

For each annotated defect on the test boards, crop the source board so the defect sits at
a growing offset from center (real board content fills the frame — no border artifacts),
score the binary good/bad model, and aggregate P(defective) vs offset. Because we crop
straight from the labelled source, the defect location is exact (no template matching).

Run it TWICE to see whether position augmentation fixes the center bias:
  * on the current model  -> expect P to fall as the defect moves off-center;
  * after retraining with `mine_patches.py --defect-offset 0.3` -> expect a flatter curve.

Usage:
  python resnet/offcenter_test.py --weights best.weights.h5 --size 256 --source hr_08
"""
import argparse, glob, os, sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import cv2
import tensorflow as tf
from resnet50_tf import build_resnet50, preprocess_batch

ROOT = Path(__file__).resolve().parent.parent


def boxes_of(img_path, W, H):
    lf = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    out = []
    if Path(lf).exists():
        for ln in Path(lf).read_text().splitlines():
            q = ln.split()
            if len(q) >= 5:
                cx, cy, bw, bh = (float(v) for v in q[1:5])
                out.append((cx * W, cy * H, bw * W, bh * H))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--patch", type=int, default=1024, help="crop window on the source board")
    ap.add_argument("--source", default="hr_08",
                    help="board filename prefix to test (e.g. a held-out template)")
    ap.add_argument("--fracs", default="0,0.1,0.2,0.3,0.4",
                    help="defect offset from center as fractions of --patch")
    ap.add_argument("--max-defects", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "resnet/offcenter_test.png"))
    args = ap.parse_args()
    fracs = [float(x) for x in args.fracs.split(",")]

    model = build_resnet50(size=args.size, freeze_backbone=True)
    model.load_weights(args.weights)

    def score(bgr):
        rgb = cv2.cvtColor(cv2.resize(bgr, (args.size, args.size), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB).astype(np.float32)
        return float(model(preprocess_batch(rgb[None]), training=False).numpy().ravel()[0])

    half = args.patch // 2
    rows, used = [], 0
    for f in sorted(glob.glob(str(ROOT / f"datasets/unified_pku_yolo/*/images/{args.source}_*"))):
        if used >= args.max_defects:
            break
        src = cv2.imread(f)
        if src is None:
            continue
        H, W = src.shape[:2]
        for dcx, dcy, bw, bh in boxes_of(f, W, H):
            if used >= args.max_defects:
                break
            room = half + int((max(fracs) + 0.05) * args.patch)
            if not (room < dcx < W - room and room < dcy < H - room):
                continue                     # need space to shift the crop and stay in-bounds
            ps = []
            for fr in fracs:
                off = int(fr * args.patch)
                ox, oy = int(dcx - half + off), int(dcy - half + off)   # defect -> top-left
                ps.append(score(src[oy:oy + args.patch, ox:ox + args.patch]))
            rows.append(ps); used += 1

    rows = np.array(rows)
    if len(rows) == 0:
        print("No defects with enough in-bounds room; try a smaller --patch or another --source.")
        return
    print(f"N = {len(rows)} defects  (source prefix '{args.source}')")
    print("offset%   meanP    caught>=0.5")
    for j, fr in enumerate(fracs):
        col = rows[:, j]
        print(f"  {int(fr*100):4d}%   {col.mean():.3f}    {int((col >= 0.5).sum())}/{len(col)}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [fr * 100 for fr in fracs]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for r in rows:
            ax.plot(xs, r, color="gray", alpha=0.2, lw=0.7)
        ax.plot(xs, rows.mean(0), color="crimson", lw=3, marker="o", label="mean P(defective)")
        ax.axhline(0.5, ls="--", color="k"); ax.set_ylim(0, 1.02)
        ax.set_xlabel("defect offset from center (% of patch)")
        ax.set_ylabel("P(defective)")
        ax.set_title(f"Off-center defect test  (N={len(rows)}, source '{args.source}')")
        ax.legend()
        fig.savefig(args.out, dpi=130, bbox_inches="tight")
        print("saved plot ->", args.out)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
