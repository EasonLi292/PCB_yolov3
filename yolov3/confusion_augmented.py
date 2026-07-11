#!/usr/bin/env python3
"""Board-level good/bad confusion for the YOLO detector with an AUGMENTED good set.

The in-domain good set is only 10 healed HRIPCB plates — far too few to estimate a false-alarm
rate (0/10 only bounds it below ~30%). Fix it the way the ResNet good set is large: turn the 10
plates into many by realistic-imaging augmentation (sensor read noise, exposure drift, focus
softness, sub-pixel registration shift, small fixture rotation), so each clean board is presented
hundreds of ways. The 1,286 defective test boards get one augmented draw each, so BOTH sides are
scored under matched realistic imaging.

Output: board-level confusion (recall, false-alarm, accuracy) at each score threshold, with a
good count comparable to the defective count.

Usage:
  python yolov3/confusion_augmented.py --ir <IR> --defective datasets/unified_pku_yolo_gray640 \
      --split test --plates "datasets/clean_plates/plate_*.png" --good-reps 128 \
      --scores 0.05 0.10 0.25 --out runs/.../confusion_augmented.json
"""
import argparse, glob, json, sys, tempfile, os
from pathlib import Path
import numpy as np, cv2
import openvino as ov

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_openvino import preprocess, postprocess


def augment(gray, rng):
    """One realistic imaging presentation of a grayscale board (ranges = a normal inspection rig)."""
    img = gray.astype(np.float32)
    sigma = rng.uniform(2.0, 6.0)                       # read noise
    img = img + rng.normal(0, sigma, img.shape)
    img = np.clip(img, 0, 255)
    g = 1.0 + rng.uniform(-0.06, 0.06)                  # exposure drift
    img = np.clip(img * g, 0, 255).astype(np.uint8)
    bs = rng.uniform(0.0, 0.8)                           # focus softness
    if bs > 0.05:
        img = cv2.GaussianBlur(img, (0, 0), bs)
    ang = rng.uniform(0, 2*np.pi); r = rng.uniform(0, 1.5)   # registration shift
    M = np.float32([[1, 0, r*np.cos(ang)], [0, 1, r*np.sin(ang)]])
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    deg = rng.uniform(-0.7, 0.7)                          # fixture rotation
    h, w = img.shape[:2]
    R = cv2.getRotationMatrix2D((w/2, h/2), deg, 1.0)
    img = cv2.warpAffine(img, R, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return img


def det_scores(compiled, fpga_raw, decode_and_nms, gray, size, min_score, iou, tmp):
    cv2.imwrite(tmp, gray)
    inp, _v, (h0, w0) = preprocess(tmp, size)
    r = compiled(inp)
    if fpga_raw:
        dets = decode_and_nms([r[o] for o in compiled.outputs], w0, h0, min_score, iou)
    else:
        dets = postprocess(r[compiled.output(0)][0], w0, h0, min_score, iou)
    return [d[4] for d in dets]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", required=True)
    ap.add_argument("--defective", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--plates", default="datasets/clean_plates/plate_*.png")
    ap.add_argument("--good-reps", type=int, default=128, help="augmented presentations per plate")
    ap.add_argument("--scores", type=float, nargs="+", default=[0.05, 0.10, 0.25])
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    compiled = ov.Core().compile_model(args.ir, "CPU")
    fpga_raw = len(compiled.outputs) >= 3
    decode_and_nms = None
    if fpga_raw:
        from yolo_postprocess import decode_and_nms
    smin = min(args.scores); rng = np.random.default_rng(args.seed)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name

    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    defective = sorted(p for p in (Path(args.defective)/args.split/"images").iterdir() if p.suffix.lower() in exts)
    plates = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in sorted(glob.glob(args.plates))]
    good_n = len(plates) * args.good_reps
    print(f"defective boards: {len(defective)} (1 augmented draw each)   "
          f"good: {len(plates)} plates x {args.good_reps} = {good_n} augmented presentations")

    # defective: each board, one augmented draw
    bad_hits = {s: 0 for s in args.scores}
    for i, p in enumerate(defective):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        scs = det_scores(compiled, fpga_raw, decode_and_nms, augment(g, rng), args.size, smin, args.iou, tmp)
        for s in args.scores:
            if any(c >= s for c in scs): bad_hits[s] += 1
        if (i+1) % 300 == 0: print(f"  defective {i+1}/{len(defective)}")

    # good: each plate augmented good_reps times
    good_hits = {s: 0 for s in args.scores}
    for j, pl in enumerate(plates):
        for _ in range(args.good_reps):
            scs = det_scores(compiled, fpga_raw, decode_and_nms, augment(pl, rng), args.size, smin, args.iou, tmp)
            for s in args.scores:
                if any(c >= s for c in scs): good_hits[s] += 1
        print(f"  plate {j+1}/{len(plates)} done")
    os.unlink(tmp)

    nD, nG = len(defective), good_n
    report = {"ir": args.ir, "n_defective": nD, "n_good_augmented": nG,
              "good_plates": len(plates), "good_reps": args.good_reps, "by_score": {}}
    print(f"\n{'score':>6} {'recall':>8} {'false-alarm':>12} {'accuracy':>10}   confusion (TP FN FP TN)")
    for s in args.scores:
        TP = bad_hits[s]; FN = nD - TP; FP = good_hits[s]; TN = nG - FP
        rec = TP/nD; fa = FP/nG; acc = (TP+TN)/(nD+nG)
        report["by_score"][f"{s:.2f}"] = {"TP":TP,"FN":FN,"FP":FP,"TN":TN,
            "recall":round(rec,4),"false_alarm":round(fa,4),"accuracy":round(acc,4),
            "precision":round(TP/(TP+FP),4) if TP+FP else 0.0}
        print(f"{s:>6.2f} {rec:>8.3f} {fa:>12.3f} {acc:>10.3f}   {TP} {FN} {FP} {TN}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2)); print("wrote", args.out)


if __name__ == "__main__":
    main()
