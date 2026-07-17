#!/usr/bin/env python3
"""Quantize the 512 ResNet good/bad classifier to FP16 / INT8 / INT4 and measure the cost.

Starting from the FP32 OpenVINO IR (the deployment artifact), produce four precisions and
evaluate each on the SAME test set, reporting accuracy, recall, disk size, and latency:

  FP32   baseline (reference)
  FP16   deterministic weight cast              (ov.save_model compress_to_fp16=True; no data)
  INT8   full post-training quantization        (NNCF: weights+activations INT8, needs calib data)
  INT4   weight-only 4-bit compression          (NNCF compress_weights INT4_SYM; activations FP)

Notes / honesty:
  * INT8 is real integer inference (weights AND activations) -> genuine speed/size win on CPU
    with VNNI and on the FPGA DLA. Calibration data is required to pick activation ranges.
  * INT4 here is WEIGHT-ONLY (the only 4-bit path OpenVINO exposes; designed for LLMs). On a
    CNN it shrinks the weights but activations stay higher precision and CPU decompresses at
    runtime, so expect an accuracy hit with little/no CPU speedup -- it's an accuracy probe,
    not a deployment recommendation. Reported for completeness.
  * Calibration uses a slice of the TRAIN or a disjoint slice of TEST; keep it out of the
    reported test metrics if you can (pass --calib-dir a train split).

Usage:
  python resnet/quantize_sweep.py --ir release_resnet_512/resnet50_pcb512_fp32.xml \
      --data release_resnet_512/test_dataset --calib 300 --outdir release_resnet_512/quant
"""
import argparse, glob, json, os, time
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
    return good + bad


def metrics(ys, ps, thr=0.5):
    ys = np.asarray(ys); ps = np.asarray(ps); pred = (ps >= thr).astype(int)
    TP = int(((pred == 1) & (ys == 1)).sum()); FN = int(((pred == 0) & (ys == 1)).sum())
    FP = int(((pred == 1) & (ys == 0)).sum()); TN = int(((pred == 0) & (ys == 0)).sum())
    rec = TP / (TP + FN) if TP + FN else 0
    prec = TP / (TP + FP) if TP + FP else 0
    acc = (TP + TN) / len(ys)
    order = np.argsort(-ps); yss = ys[order]; P = ys.sum(); N = len(ys) - P
    auc = float(np.trapz(np.cumsum(yss) / P, np.cumsum(1 - yss) / N)) if P and N else 0
    # best-threshold accuracy: quantization can shift a sigmoid's operating point, so the
    # fixed 0.5 threshold is unfair across precisions -- report the achievable accuracy too.
    best_acc, best_thr = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, 19):
        a = float(((ps >= t).astype(int) == ys).mean())
        if a > best_acc:
            best_acc, best_thr = a, float(t)
    return dict(TP=TP, FN=FN, FP=FP, TN=TN, recall=round(rec, 4), precision=round(prec, 4),
                accuracy=round(acc, 4), roc_auc=round(auc, 4),
                acc_best=round(best_acc, 4), thr_best=round(best_thr, 2))


def evaluate(compiled, files, size):
    out = compiled.output(0)
    ys, ps, t0 = [], [], time.time()
    for p, y in files:
        ps.append(float(compiled(preprocess(p, size))[out].ravel()[0])); ys.append(y)
    lat = (time.time() - t0) / len(files) * 1000
    m = metrics(ys, ps); m["latency_ms"] = round(lat, 2)
    return m, np.array(ps)


def bin_size_mb(xml_path):
    b = Path(xml_path).with_suffix(".bin")
    return round(b.stat().st_size / 1e6, 1) if b.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", required=True, help="FP32 OpenVINO IR .xml")
    ap.add_argument("--data", required=True, help="test dir with good/ bad/")
    ap.add_argument("--calib-dir", default="", help="calibration images dir (defaults to --data)")
    ap.add_argument("--calib", type=int, default=300, help="# calibration samples for INT8")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="limit test images (0=all)")
    ap.add_argument("--outdir", default="quant_out")
    ap.add_argument("--device", default="CPU")
    args = ap.parse_args()

    core = ov.Core()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    files = load_files(args.data)
    if args.limit:
        files = files[:args.limit // 2] + files[-args.limit // 2:]
    print(f"test images: {len(files)}   device: {args.device}")

    fp32 = core.read_model(args.ir)
    results = {}

    # ---- FP32 baseline ----
    print("\n[FP32] baseline ...")
    m, _ = evaluate(core.compile_model(fp32, args.device), files, args.size)
    m["size_mb"] = bin_size_mb(args.ir); results["FP32"] = m

    # ---- FP16 (weight cast, no data) ----
    print("[FP16] compress_to_fp16 ...")
    fp16_xml = outdir / "resnet50_fp16.xml"
    ov.save_model(fp32, str(fp16_xml), compress_to_fp16=True)
    m, _ = evaluate(core.compile_model(core.read_model(str(fp16_xml)), args.device), files, args.size)
    m["size_mb"] = bin_size_mb(fp16_xml); results["FP16"] = m

    # ---- INT8 (full PTQ) and INT4 (weight-only) need NNCF ----
    try:
        import nncf
        calib_dir = args.calib_dir or args.data
        calib_files = load_files(calib_dir)
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(calib_files))[:args.calib]
        calib_paths = [calib_files[i][0] for i in idx]
        print(f"[INT8] NNCF PTQ with {len(calib_paths)} calibration images ...")
        calib_ds = nncf.Dataset(calib_paths, lambda p: preprocess(p, args.size))
        # MIXED preset = asymmetric activation ranges; PERFORMANCE (symmetric) can shift a
        # sigmoid classifier's operating point and collapse it to one class.
        int8 = nncf.quantize(fp32, calib_ds, subset_size=len(calib_paths),
                             preset=nncf.QuantizationPreset.MIXED)
        int8_xml = outdir / "resnet50_int8.xml"; ov.save_model(int8, str(int8_xml))
        m, _ = evaluate(core.compile_model(int8, args.device), files, args.size)
        m["size_mb"] = bin_size_mb(int8_xml); results["INT8"] = m
    except Exception as e:
        results["INT8"] = {"error": str(e)[:200]}
        print("  INT8 failed:", e)

    try:
        import nncf
        print("[INT4] NNCF weight-only compress_weights (INT4_SYM, all_layers) ...")
        try:
            int4 = nncf.compress_weights(fp32, mode=nncf.CompressWeightsMode.INT4_SYM,
                                         ratio=1.0, group_size=64, all_layers=True)
        except Exception:
            int4 = nncf.compress_weights(fp32, mode=nncf.CompressWeightsMode.INT4_SYM)
        int4_xml = outdir / "resnet50_int4.xml"; ov.save_model(int4, str(int4_xml))
        m, _ = evaluate(core.compile_model(int4, args.device), files, args.size)
        m["size_mb"] = bin_size_mb(int4_xml); results["INT4"] = m
    except Exception as e:
        results["INT4"] = {"error": str(e)[:200]}
        print("  INT4 failed:", e)

    # ---- report ----
    (outdir / "quant_sweep.json").write_text(json.dumps(results, indent=2))
    base = results["FP32"]
    print("\n===================== QUANTIZATION SWEEP =====================")
    print(f"{'prec':5s} {'size_MB':>8s} {'lat_ms':>7s} {'AUC':>7s} {'acc@.5':>7s} "
          f"{'acc@best':>9s} {'thr*':>5s} {'recall@.5':>9s}  ΔAUC")
    for k in ["FP32", "FP16", "INT8", "INT4"]:
        r = results.get(k, {})
        if "error" in r:
            print(f"{k:5s}  -> ERROR: {r['error']}"); continue
        dauc = r["roc_auc"] - base["roc_auc"]
        print(f"{k:5s} {str(r['size_mb']):>8s} {r['latency_ms']:>7.1f} {r['roc_auc']:>7.3f} "
              f"{r['accuracy']:>7.3f} {r['acc_best']:>9.3f} {r['thr_best']:>5.2f} "
              f"{r['recall']:>9.3f}  {dauc:+.3f}")
    print("\nAUC is threshold-independent (the fair cross-precision accuracy); acc@best shows")
    print("the achievable accuracy after re-tuning the decision threshold per precision.")
    print(f"\nwrote {outdir/'quant_sweep.json'}")


if __name__ == "__main__":
    main()
