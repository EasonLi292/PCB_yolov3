#!/usr/bin/env python3
"""Confusion matrices for 5 weight numeric formats, all with FULL-PRECISION activations.

Isolates the weight-format effect (activations + accumulate stay FP32 = the FPGA block-float
datapath). Every large weight tensor is cast uniformly to the format, then the full test set is
evaluated. Formats:

  INT4  per-output-channel symmetric 4-bit integer
  INT8  per-output-channel symmetric 8-bit integer
  FP11  float: 1 sign + 5 exponent + 5 mantissa   (Intel DLA FP11)
  FP13  float: 1 sign + 5 exponent + 7 mantissa   (Agilex FP13AGX)
  FP16  float: 1 sign + 5 exponent + 10 mantissa  (IEEE half)

The three FP formats share FP16's 5-bit exponent (same dynamic range) and differ only in mantissa
precision, so this is simulated by rounding each weight's mantissa to {5,7,10} bits.

Usage:
  python resnet/weight_format_confusion.py --ir release_resnet_512/resnet50_pcb512_fp32.xml \
      --data release_resnet_512/test_dataset --outdir release_resnet_512/weight_formats
"""
import argparse, glob, json, shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import cv2, openvino as ov
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

MEAN = np.array([103.939, 116.779, 123.68], np.float32)


def preprocess(p, size=512):
    img = cv2.imread(p); img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.ascontiguousarray((img[..., ::-1] - MEAN)[None])


def load_files(d):
    g = [(p, 0) for p in sorted(glob.glob(f"{d}/good/*")) if Path(p).is_file()]
    b = [(p, 1) for p in sorted(glob.glob(f"{d}/bad/*")) if Path(p).is_file()]
    return g + b


def expose_logit(m):
    for op in m.get_ordered_ops():
        if op.get_type_name() == "Sigmoid":
            m.add_outputs([op.input_value(0)]); return True
    return False


def list_weights(xml, min_elems=256):
    root = ET.parse(xml).getroot(); out = []
    for layer in root.iter("layer"):
        if layer.get("type") != "Const":
            continue
        d = layer.find("data")
        if d is None or d.get("element_type") != "f32":
            continue
        shape = [int(s) for s in d.get("shape").split(",") if s.strip()]
        nel = int(np.prod(shape)) if shape else 0
        if len(shape) < 2 or nel < min_elems:
            continue
        out.append((shape, int(d.get("offset")), int(d.get("size"))))
    return out


def q_int(w, bits):
    qmax = (1 << (bits - 1)) - 1
    flat = w.reshape(w.shape[0], -1)
    scale = np.abs(flat).max(axis=1, keepdims=True) / qmax
    scale[scale == 0] = 1.0
    return (np.clip(np.round(flat / scale), -qmax, qmax) * scale).reshape(w.shape).astype(np.float32)


def q_float(w, mantissa_bits):
    """Round to `mantissa_bits` mantissa bits (shared FP16-style 5-bit exponent kept intact)."""
    m, e = np.frexp(w)                       # w = m * 2**e, |m| in [0.5,1)
    step = 1 << (mantissa_bits + 1)
    return np.ldexp(np.round(m * step) / step, e).astype(np.float32)


FORMATS = {
    "INT4": lambda w: q_int(w, 4),
    "INT8": lambda w: q_int(w, 8),
    "FP11": lambda w: q_float(w, 5),
    "FP13": lambda w: q_float(w, 7),
    "FP16": lambda w: q_float(w, 10),
}
BITS = {"INT4": 4, "INT8": 8, "FP11": 11, "FP13": 13, "FP16": 16}


def write_format(base_buf, weights, fn, out_bin):
    buf = bytearray(base_buf)
    for shape, off, size in weights:
        w = np.frombuffer(bytes(buf[off:off + size]), np.float32).reshape(shape).copy()
        buf[off:off + size] = fn(w).tobytes()
    Path(out_bin).write_bytes(bytes(buf))


def evaluate(core, xml, files, size):
    m = core.read_model(str(xml)); hl = expose_logit(m)
    comp = core.compile_model(m, "CPU"); op_p = comp.output(0)
    op_l = comp.output(1) if hl and len(comp.outputs) > 1 else None
    ps, ls = [], []
    for p, _ in files:
        r = comp(preprocess(p, size)); ps.append(float(r[op_p].ravel()[0]))
        ls.append(float(r[op_l].ravel()[0]) if op_l is not None else np.nan)
    return np.array(ps), np.array(ls)


def confusion(ys, score, thr):
    pred = (np.asarray(score) >= thr).astype(int)
    TP = int(((pred == 1) & (ys == 1)).sum()); FN = int(((pred == 0) & (ys == 1)).sum())
    FP = int(((pred == 1) & (ys == 0)).sum()); TN = int(((pred == 0) & (ys == 0)).sum())
    return dict(TP=TP, FN=FN, FP=FP, TN=TN, accuracy=round((TP + TN) / len(ys), 4),
                recall=round(TP / (TP + FN + 1e-9), 4), precision=round(TP / (TP + FP + 1e-9), 4))


def safe_auc(y, s):
    s = np.asarray(s, float); m = np.isfinite(s)
    return round(float(roc_auc_score(y[m], s[m])), 4) if m.sum() > 1 and len(np.unique(y[m])) > 1 else float("nan")


def best_logit(ys, ls):
    ls = np.nan_to_num(ls, nan=0.0)
    grid = np.linspace(ls.min(), ls.max(), 201)
    a, t = max((float(((ls >= g).astype(int) == ys).mean()), float(g)) for g in grid)
    return round(a, 4), round(t, 3)


def plot_cm(c, title, path):
    fig, ax = plt.subplots(figsize=(3.9, 3.9))
    cm = np.array([[c["TN"], c["FP"]], [c["FN"], c["TP"]]])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred good", "pred bad"], fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["good", "bad"], fontsize=8)
    ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--outdir", default="weight_formats")
    args = ap.parse_args()

    core = ov.Core()
    out = Path(args.outdir); (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)
    base_buf = Path(args.ir).with_suffix(".bin").read_bytes()
    weights = list_weights(args.ir)
    files = load_files(args.data)
    if args.limit:
        files = files[:args.limit // 2] + files[-args.limit // 2:]
    ys = np.array([y for _, y in files])
    print(f"weight tensors {len(weights)}   test images {len(ys)}")

    results = {}
    # FP32 baseline for context
    for name in ["FP32", "INT4", "INT8", "FP11", "FP13", "FP16"]:
        xml = out / "models" / f"{name}.xml"; shutil.copy(args.ir, xml)
        if name == "FP32":
            shutil.copy(Path(args.ir).with_suffix(".bin"), xml.with_suffix(".bin"))
        else:
            write_format(base_buf, weights, FORMATS[name], xml.with_suffix(".bin"))
        ps, ls = evaluate(core, xml, files, args.size)
        c = confusion(ys, ps, 0.5)
        c["auc"] = safe_auc(ys, ls)
        c["acc_best_logit"], c["thr_best_logit"] = best_logit(ys, ls)
        c["bits"] = BITS.get(name, 32)
        results[name] = c
        note = "" if name == "FP32" else f" [{name}, activations FP]"
        plot_cm(c, f"{name}  (weights {c['bits']}-bit, act FP)\n"
                   f"acc {c['accuracy']:.3f} @0.5 / {c['acc_best_logit']:.3f} @best  AUC {c['auc']:.3f}",
                out / "figures" / f"confusion_{name}.png")
        print(f"[{name:5s}] {c['bits']:>2d}b  acc@.5 {c['accuracy']:.4f}  acc@best {c['acc_best_logit']:.4f}  "
              f"AUC {c['auc']:.4f}  (TN{c['TN']} FP{c['FP']} FN{c['FN']} TP{c['TP']})")

    # combined grid
    order = ["INT4", "INT8", "FP11", "FP13", "FP16"]
    fig, axes = plt.subplots(1, 5, figsize=(17.5, 3.9))
    for ax, k in zip(axes, order):
        c = results[k]; cm = np.array([[c["TN"], c["FP"]], [c["FN"], c["TP"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred good", "pred bad"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["good", "bad"], fontsize=8)
        ax.set_title(f"{k} ({c['bits']}-bit weights)\nacc {c['accuracy']:.3f}/{c['acc_best_logit']:.3f}  "
                     f"AUC {c['auc']:.3f}", fontsize=8.5)
    fig.suptitle("Confusion matrices by weight format (activations full precision)  — "
                 f"FP32 ref: acc {results['FP32']['accuracy']:.3f}", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "figures" / "confusion_weight_formats.png", dpi=130); plt.close(fig)

    (out / "weight_formats.json").write_text(json.dumps(results, indent=2))
    print(f"\nfigures -> {out/'figures'}/   json -> {out/'weight_formats.json'}")


if __name__ == "__main__":
    main()
