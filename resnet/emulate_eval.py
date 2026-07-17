#!/usr/bin/env python3
"""Evaluate the 512 ResNet good/bad IR through the OpenVINO Python API and dump per-image
outputs -- the reference that the FPGA AI Suite software emulator is compared against.

Two modes, SAME code path (so CPU vs emulator differ only in the compute device):

  --device CPU                 reference run (any box with OpenVINO). Do this FIRST.
  --device HETERO:FPGA,CPU     bit-accurate FPGA AI Suite emulator (needs the AI Suite
                               container + an .arch file passed via --arch). The DLA runs
                               block-floating-point, so expect ~FP16 numbers, not FP32.

Verification workflow:
  1) run --device CPU  --dump ref_cpu.npz          (here / this Mac)
  2) run --device HETERO:FPGA,CPU --arch X.arch --dump emu.npz   (Linux box, in container)
  3) compare: np.load both, check max|prob_cpu - prob_emu| and that class decisions agree.

Also exposes the PRE-SIGMOID LOGIT (by adding the Sigmoid's input as a second model output)
so we can threshold in logit space -- robust to the sigmoid saturating under quantization/BFP.

Usage:
  python resnet/emulate_eval.py --ir release_resnet_512/resnet50_pcb512_fp32.xml \
      --data release_resnet_512/test_dataset --device CPU --dump release_resnet_512/emu/ref_cpu.npz
"""
import argparse, glob, time
from pathlib import Path
import numpy as np
import cv2
import openvino as ov

MEAN = np.array([103.939, 116.779, 123.68], np.float32)  # keras caffe BGR means


def preprocess(path, size):
    img = cv2.imread(path)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = img[..., ::-1] - MEAN                          # RGB->BGR, subtract means
    return np.ascontiguousarray(img[None])               # [1,size,size,3]


def load_files(data_dir):
    good = [(p, 0) for p in sorted(glob.glob(f"{data_dir}/good/*"))]
    bad = [(p, 1) for p in sorted(glob.glob(f"{data_dir}/bad/*"))]
    return [(p, y) for p, y in good + bad if Path(p).is_file()]


def metrics(ys, ps, thr=0.5):
    ys = np.asarray(ys); ps = np.asarray(ps); pred = (ps >= thr).astype(int)
    TP = int(((pred == 1) & (ys == 1)).sum()); FN = int(((pred == 0) & (ys == 1)).sum())
    FP = int(((pred == 1) & (ys == 0)).sum()); TN = int(((pred == 0) & (ys == 0)).sum())
    rec = TP / (TP + FN) if TP + FN else 0
    prec = TP / (TP + FP) if TP + FP else 0
    acc = (TP + TN) / len(ys)
    order = np.argsort(-ps); yss = ys[order]; P = ys.sum(); N = len(ys) - P
    auc = float(np.trapz(np.cumsum(yss) / P, np.cumsum(1 - yss) / N)) if P and N else 0
    return dict(TP=TP, FN=FN, FP=FP, TN=TN, recall=round(rec, 4),
                precision=round(prec, 4), accuracy=round(acc, 4), roc_auc=round(auc, 4))


def best_threshold(ys, ps, lo, hi):
    ys = np.asarray(ys)
    best_a, best_t = 0.0, lo
    for t in np.linspace(lo, hi, 41):
        a = float(((np.asarray(ps) >= t).astype(int) == ys).mean())
        if a > best_a:
            best_a, best_t = a, float(t)
    return round(best_a, 4), round(best_t, 4)


def expose_logit(model):
    """Add the pre-sigmoid tensor as a second model output. Returns True if a Sigmoid was found."""
    for op in model.get_ordered_ops():
        if op.get_type_name() == "Sigmoid":
            src = op.input_value(0)          # the logit feeding the sigmoid
            model.add_outputs([src])
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--device", default="CPU",
                    help="CPU (reference) or HETERO:FPGA,CPU (AI Suite emulator)")
    ap.add_argument("--arch", default="", help=".arch file (required for the FPGA emulator)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="limit test images (0=all)")
    ap.add_argument("--dump", default="", help="write per-image outputs to this .npz")
    args = ap.parse_args()

    core = ov.Core()
    print("openvino", ov.__version__, "| devices", core.available_devices, "| target", args.device)
    model = core.read_model(args.ir)
    has_logit = expose_logit(model)
    print("pre-sigmoid logit exposed:", has_logit)

    cfg = {}
    if args.arch:
        cfg["ARCH_PATH"] = args.arch          # AI Suite emulation/FPGA plugin config key
    compiled = core.compile_model(model, args.device, cfg)
    out_prob = compiled.output(0)
    out_logit = compiled.output(1) if has_logit and len(compiled.outputs) > 1 else None

    files = load_files(args.data)
    if args.limit:
        files = files[:args.limit // 2] + files[-args.limit // 2:]
    print(f"test images: {len(files)}")

    ys, ps, ls, paths, t0 = [], [], [], [], time.time()
    for i, (p, y) in enumerate(files):
        r = compiled(preprocess(p, args.size))
        ps.append(float(r[out_prob].ravel()[0]))
        ls.append(float(r[out_logit].ravel()[0]) if out_logit is not None else np.nan)
        ys.append(y); paths.append(p)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}")
    lat = (time.time() - t0) / len(files) * 1000
    ys = np.array(ys); ps = np.array(ps); ls = np.array(ls)

    print("\n================= EMULATE/REFERENCE EVAL =================")
    print(f"device {args.device}   latency {lat:.1f} ms/img")
    m = metrics(ys, ps); ab, tb = best_threshold(ys, ps, 0.05, 0.95)
    print(f"[sigmoid prob] @0.5  acc {m['accuracy']:.4f}  recall {m['recall']:.4f}  "
          f"prec {m['precision']:.4f}  AUC {m['roc_auc']:.4f}  |  best acc {ab:.4f} @ thr {tb:.3f}")
    print(f"               TP {m['TP']}  FN {m['FN']}  FP {m['FP']}  TN {m['TN']}")
    if out_logit is not None:
        ml = metrics(ys, ls, thr=0.0); alb, tlb = best_threshold(ys, ls, ls.min(), ls.max())
        print(f"[logit]  sign(>0)  acc {ml['accuracy']:.4f}  recall {ml['recall']:.4f}  "
              f"prec {ml['precision']:.4f}  |  best acc {alb:.4f} @ logit {tlb:.3f}")
        print(f"         logit range [{ls.min():.3f}, {ls.max():.3f}]  "
              f"prob range [{ps.min():.5f}, {ps.max():.5f}]")

    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.dump, y=ys, prob=ps, logit=ls,
                 path=np.array(paths), device=args.device, latency_ms=lat)
        print(f"\nwrote {args.dump}  (diff CPU vs emulator: "
              f"np.max(np.abs(a['prob']-b['prob'])) should be ~1e-2 or less)")


if __name__ == "__main__":
    main()
