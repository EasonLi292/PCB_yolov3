#!/usr/bin/env python3
"""Stratified detection rate vs defect offset — OFFSET-TRAINED models, 256 and 512.

Question: on the deployment (offset-trained) model, are defects that land FURTHER from the
tile center caught LESS often? The aggregate offset-test recall (0.933@256 / 0.985@512)
hides this — it averages over all offsets. Here we resolve recall as a function of the
defect's actual offset.

Method (causal, on held-in TEST templates):
  * take each annotated defect on the test-split boards (default hr_01, hr_04),
  * draw K random placements from the SAME distribution training used
    (offset ~ U[-0.4,0.4]*patch on each axis -> the defect lands anywhere in the tile),
  * crop the real board so the defect sits at that offset (real content fills the frame,
    no border artifacts), score P(defective) with the offset-trained model,
  * bin every (defect, placement) sample by its offset magnitude = max(|ox|,|oy|)/patch
    (Chebyshev -> "within X% of center", bounded [0,0.4]) and report recall per bin.

Runs BOTH offset-trained models (256, 512) and writes a combined table + plot + JSON.

Usage (on the GPU box, weights present):
  python resnet/offset_recall_stratified.py \
      --w256 runs_resnet_v3/pcb_bin_offset_256/best.weights.h5 \
      --w512 runs_resnet_v3/pcb_bin_offset_512/best.weights.h5 \
      --sources hr_01,hr_04 --k 6 --thr 0.5
"""
import argparse, glob, json, os, sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import cv2
import tensorflow as tf
from resnet50_tf import build_resnet50, preprocess_batch

ROOT = Path(__file__).resolve().parent.parent
BINS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4)]


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


def collect_samples(size, weights, sources, patch, k, thr, seed, max_defects):
    """Return list of (offset_frac, P, caught) for one model."""
    rng = np.random.default_rng(seed)
    model = build_resnet50(size=size, freeze_backbone=True)
    model.load_weights(weights)
    half = patch // 2
    off_max = 0.4 * patch
    room = half + int(off_max + 0.05 * patch)

    def score(bgr):
        rgb = cv2.cvtColor(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB).astype(np.float32)
        return float(model(preprocess_batch(rgb[None]), training=False).numpy().ravel()[0])

    samples, used = [], 0
    files = []
    for pre in sources:
        files += sorted(glob.glob(str(ROOT / f"datasets/unified_pku_yolo/*/images/{pre}_*")))
    for f in files:
        if used >= max_defects:
            break
        src = cv2.imread(f)
        if src is None:
            continue
        H, W = src.shape[:2]
        for dcx, dcy, bw, bh in boxes_of(f, W, H):
            if used >= max_defects:
                break
            if not (room < dcx < W - room and room < dcy < H - room):
                continue                       # need room to shift the crop and stay in-bounds
            # one centered reference + K random placements from the training distribution
            placements = [(0.0, 0.0)] + [(rng.uniform(-off_max, off_max),
                                          rng.uniform(-off_max, off_max)) for _ in range(k)]
            for ox, oy in placements:
                x0, y0 = int(dcx - half - ox), int(dcy - half - oy)   # defect -> offset from center
                p = score(src[y0:y0 + patch, x0:x0 + patch])
                offfrac = max(abs(ox), abs(oy)) / patch
                samples.append((offfrac, p, int(p >= thr)))
            used += 1
    return samples, used


def summarize(samples):
    a = np.array(samples)                      # cols: offfrac, P, caught
    rows = []
    for lo, hi in BINS:
        m = (a[:, 0] >= lo) & (a[:, 0] < hi)
        n = int(m.sum())
        rows.append({"bin": f"{int(lo*100)}-{int(hi*100)}%", "n": n,
                     "recall": float(a[m, 2].mean()) if n else None,
                     "meanP": float(a[m, 1].mean()) if n else None})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w256", required=True, help="offset-trained 256 weights")
    ap.add_argument("--w512", required=True, help="offset-trained 512 weights")
    ap.add_argument("--sources", default="hr_01,hr_04",
                    help="test-split board prefixes (see dataset_manifest template_split)")
    ap.add_argument("--patch", type=int, default=1024)
    ap.add_argument("--k", type=int, default=6, help="random placements sampled per defect")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-defects", type=int, default=400)
    ap.add_argument("--out-json", default=str(ROOT / "resnet/details/offset_recall_stratified.json"))
    ap.add_argument("--out-png", default=str(ROOT / "resnet/figures/offset_recall_stratified.png"))
    args = ap.parse_args()
    sources = [s.strip() for s in args.sources.split(",")]

    result = {"sources": sources, "patch": args.patch, "k": args.k, "thr": args.thr,
              "offset_metric": "max(|ox|,|oy|)/patch (Chebyshev)", "models": {}}
    for tag, size, w in [("256", 256, args.w256), ("512", 512, args.w512)]:
        print(f"\n==== offset-trained {tag} model : {w} ====")
        samples, n_def = collect_samples(size, w, sources, args.patch, args.k,
                                         args.thr, args.seed, args.max_defects)
        rows = summarize(samples)
        result["models"][tag] = {"weights": w, "n_defects": n_def,
                                 "n_samples": len(samples), "bins": rows}
        print(f"  defects used: {n_def}   samples: {len(samples)}")
        print(f"  {'offset bin':10s} {'n':>6s} {'recall':>8s} {'meanP':>8s}")
        for r in rows:
            rc = "—" if r["recall"] is None else f"{r['recall']:.3f}"
            mp = "—" if r["meanP"] is None else f"{r['meanP']:.3f}"
            print(f"  {r['bin']:10s} {r['n']:6d} {rc:>8s} {mp:>8s}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print("\nwrote", args.out_json)

    # markdown table (paste into the report)
    print("\n---- markdown ----")
    print(f"| offset bin | 256 recall | 256 meanP | 512 recall | 512 meanP | n (per model) |")
    print(f"|---|---|---|---|---|---|")
    r256, r512 = result["models"]["256"]["bins"], result["models"]["512"]["bins"]
    for a, b in zip(r256, r512):
        f = lambda v: "—" if v is None else f"{v:.3f}"
        print(f"| {a['bin']} | {f(a['recall'])} | {f(a['meanP'])} | "
              f"{f(b['recall'])} | {f(b['meanP'])} | {a['n']} |")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [f"{int(lo*100)}-{int(hi*100)}" for lo, hi in BINS]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for tag, c in [("256", "tab:blue"), ("512", "tab:red")]:
            ys = [r["recall"] for r in result["models"][tag]["bins"]]
            ax.plot(xs, ys, marker="o", lw=2.5, color=c, label=f"{tag}·offset")
        ax.set_ylim(0, 1.02); ax.axhline(0.9, ls="--", color="gray", lw=1)
        ax.set_xlabel("defect offset from tile center (% of patch)")
        ax.set_ylabel(f"recall (P >= {args.thr})")
        ax.set_title("Detection rate vs defect offset — offset-trained models")
        ax.legend()
        Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_png, dpi=130, bbox_inches="tight")
        print("saved plot ->", args.out_png)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
