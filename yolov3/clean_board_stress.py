#!/usr/bin/env python3
"""
Clean-board FALSE-ALARM stress test for the YOLOv3 detector.

The in-domain good set is only the 10 healed HRIPCB clean plates -- HRIPCB has 10 board
designs and never a defect-free photo, so 0/10 flagged only bounds the true false-alarm
rate below ~30% (rule of three). Useless for a line spec.

This turns those 10 boards into hundreds of clean *presentations*: image each plate many
times under realistic sensor variation (read noise, exposure drift, focus softness,
registration shift, fixture rotation) and count how often the detector hallucinates a
defect on a board that is, by construction, perfectly clean. That is the number a
production line actually cares about, and it localizes WHERE (which nuisance, what
magnitude) the false alarms begin.

Inference path is byte-identical to board_level_eval.py / analyze_openvino.py (the same
export_openvino.preprocess grayscale->resize->IR), so the magnitude-0 row must reproduce
the reported 0/10.

Usage:
  python yolov3/clean_board_stress.py --ir <run>/openvino_fpga/yolov3_fpga_fp32.xml \
      --plates "datasets/clean_plates/plate_*.png" --size 640 --reps 8
"""
import argparse, glob, sys, json, tempfile, os
from pathlib import Path
import numpy as np, cv2
import openvino as ov

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_openvino import preprocess, postprocess


# ---- realistic imaging nuisances, applied to the grayscale plate the model actually sees ----
def noise(img, sigma, rng):                 # sensor read noise (gray levels)
    if sigma == 0: return img
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)

def gain(img, pct, rng):                    # exposure / lighting drift (multiplicative, random sign)
    if pct == 0: return img
    s = pct * (1 if rng.random() < 0.5 else -1)
    return np.clip(img.astype(np.float32) * (1.0 + s), 0, 255).astype(np.uint8)

def blur(img, sigma, rng):                  # focus softness
    return cv2.GaussianBlur(img, (0, 0), sigma) if sigma > 0 else img

def shift(img, px, rng):                     # sub-pixel registration error (random direction)
    if px == 0: return img
    ang = rng.random() * 2 * np.pi
    M = np.float32([[1, 0, px * np.cos(ang)], [0, 1, px * np.sin(ang)]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def rot(img, deg, rng):                      # fixture rotation error (random sign)
    if deg == 0: return img
    d = deg * (1 if rng.random() < 0.5 else -1)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), d, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

SWEEPS = {
    "noise (sigma gray levels)": (noise, [0, 2, 5, 10, 15, 20]),
    "gain (exposure drift)":     (gain,  [0.0, 0.05, 0.10, 0.20, 0.30]),
    "blur (gaussian sigma px)":  (blur,  [0, 0.5, 1.0, 1.5, 2.0]),
    "shift (px registration)":   (shift, [0, 1, 2, 4, 8]),
    "rotation (degrees)":        (rot,   [0, 0.5, 1.0, 2.0, 4.0]),
}


def det_scores(compiled, fpga_raw, decode_and_nms, gray_img, size, min_score, iou, tmp):
    """Run detector ONCE at the lowest score threshold; return the score of every surviving
    detection. Higher thresholds are then just a Python filter (NMS keeps the top box per
    cluster, so post-filtering is equivalent to '>=1 detection above s'). Writes a temp PNG
    so preprocess reads it identically to a deployed board."""
    cv2.imwrite(tmp, gray_img)
    inp, _vis, (h0, w0) = preprocess(tmp, size)
    result = compiled(inp)
    if fpga_raw:
        raw = [result[o] for o in compiled.outputs]
        dets = decode_and_nms(raw, w0, h0, min_score, iou)
    else:
        pred = result[compiled.output(0)][0]
        dets = postprocess(pred, w0, h0, min_score, iou)
    return [d[4] for d in dets]          # detection confidence scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", required=True)
    ap.add_argument("--plates", default="datasets/clean_plates/plate_*.png")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--scores", type=float, nargs="+", default=[0.10, 0.25])
    ap.add_argument("--reps", type=int, default=8, help="randomized presentations per plate per magnitude")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    compiled = ov.Core().compile_model(args.ir, "CPU")
    fpga_raw = len(compiled.outputs) >= 3
    decode_and_nms = None
    if fpga_raw:
        from yolo_postprocess import decode_and_nms
    print(f"IR outputs: {len(compiled.outputs)} ->",
          "FPGA raw heads (host decode)" if fpga_raw else "decoded IR")

    plate_paths = sorted(glob.glob(args.plates))
    plates = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in plate_paths]
    print(f"clean plates: {len(plates)}   reps/plate/magnitude: {args.reps}   scores: {args.scores}\n")
    rng = np.random.default_rng(0)
    tmpf = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    smin = min(args.scores)

    report = {"ir": args.ir, "n_plates": len(plates), "reps": args.reps, "sweeps": {}}
    for title, (fn, mags) in SWEEPS.items():
        print(f"== {title} ==")
        header = "  mag      presentations   " + "   ".join(f"flag@{s:.2f}" for s in args.scores)
        print(header)
        report["sweeps"][title] = []
        for m in mags:
            reps = 1 if m == 0 else args.reps          # magnitude 0 is deterministic
            flags = {s: 0 for s in args.scores}
            total = 0
            for img in plates:
                for _ in range(reps):
                    pert = fn(img, m, rng)
                    total += 1
                    scs = det_scores(compiled, fpga_raw, decode_and_nms, pert, args.size, smin, args.iou, tmpf)
                    for s in args.scores:
                        if any(c >= s for c in scs):
                            flags[s] += 1
            row = {"magnitude": m, "presentations": total,
                   "flag_rate": {f"{s:.2f}": flags[s] / total for s in args.scores},
                   "flagged": {f"{s:.2f}": flags[s] for s in args.scores}}
            report["sweeps"][title].append(row)
            cells = "   ".join(f"{flags[s]/total:6.3f}" for s in args.scores)
            print(f"  {m:<7}  {total:8d}          {cells}")
        print()

    # combined "realistic single frame": all nuisances at once, moderate magnitudes
    print("== combined realistic frame (noise5 + gain5% + blur0.5 + shift1px + rot0.5deg) ==")
    flags = {s: 0 for s in args.scores}; total = 0
    for img in plates:
        for _ in range(args.reps):
            p = noise(img, 5, rng); p = gain(p, 0.05, rng); p = blur(p, 0.5, rng)
            p = shift(p, 1, rng); p = rot(p, 0.5, rng)
            total += 1
            scs = det_scores(compiled, fpga_raw, decode_and_nms, p, args.size, smin, args.iou, tmpf)
            for s in args.scores:
                if any(c >= s for c in scs):
                    flags[s] += 1
    report["combined_realistic"] = {"presentations": total,
        "flag_rate": {f"{s:.2f}": flags[s] / total for s in args.scores},
        "flagged": {f"{s:.2f}": flags[s] for s in args.scores}}
    for s in args.scores:
        print(f"  flag rate @ {s:.2f}: {flags[s]/total:.3f}  ({flags[s]}/{total})")

    os.unlink(tmpf)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
