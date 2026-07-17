#!/usr/bin/env python3
"""Sensitivity-guided MIXED-precision weight quantization for the 512 ResNet good/bad classifier.

Motivation: uniform INT8 (weights+activations) collapses this subtle-defect model, and uniform
4-bit weights drop to ~0.65. But layers are not equally fragile. This does the sensible thing:

  1) SENSITIVITY: quantize each weight tensor alone to 4-bit (rest untouched), measure the damage
     as MSE of the pre-sigmoid logit vs the FP32 reference on a small balanced subset.
  2) ALLOCATE: give the K most-sensitive tensors 8-bit, everything else 4-bit. Activations and
     accumulation stay high-precision (dequant->FP compute), matching the FPGA block-float datapath
     (FP32 accumulator). This is weight-memory compression, which is the part worth saving.
  3) TRACE: sweep K, evaluate on the full test set, and report accuracy vs average weight-bits
     (a memory/resource axis). Record the confusion matrix at the chosen operating point.

Quantizer: per-output-channel symmetric N-bit, applied by editing the IR .bin directly.

Usage:
  python resnet/mixed_precision.py --ir release_resnet_512/resnet50_pcb512_fp32.xml \
      --data release_resnet_512/test_dataset --outdir release_resnet_512/mixed_precision
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
    """Return [(idx, name, shape, offset, size)] for the large f32 weight Consts (in xml order)."""
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
        out.append((len(out), layer.get("name"), shape, int(d.get("offset")), int(d.get("size"))))
    return out


def quant_channel(w, bits):
    if bits >= 16:
        return w
    qmax = (1 << (bits - 1)) - 1
    flat = w.reshape(w.shape[0], -1)
    scale = np.abs(flat).max(axis=1, keepdims=True) / qmax
    scale[scale == 0] = 1.0
    q = np.clip(np.round(flat / scale), -qmax, qmax) * scale
    return q.reshape(w.shape).astype(np.float32)


def write_mixed(base_buf, weights, assign, out_bin):
    """assign: dict idx->bits. Idx not in assign stays original f32."""
    buf = bytearray(base_buf)
    for idx, name, shape, off, size in weights:
        bits = assign.get(idx, 16)
        if bits >= 16:
            continue
        w = np.frombuffer(bytes(buf[off:off + size]), np.float32).reshape(shape).copy()
        buf[off:off + size] = quant_channel(w, bits).tobytes()
    Path(out_bin).write_bytes(bytes(buf))


def eval_model(core, xml, files, size, want_logit=True):
    m = core.read_model(str(xml)); hl = expose_logit(m)
    comp = core.compile_model(m, "CPU"); op_p = comp.output(0)
    op_l = comp.output(1) if hl and want_logit and len(comp.outputs) > 1 else None
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
    return float(roc_auc_score(y[m], s[m])) if m.sum() > 1 and len(np.unique(y[m])) > 1 else float("nan")


def avg_bits(weights, assign):
    num = sum(int(np.prod(sh)) * assign.get(i, 16) for i, _, sh, _, _ in weights)
    den = sum(int(np.prod(sh)) for i, _, sh, _, _ in weights)
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--sens-n", type=int, default=128, help="balanced subset for sensitivity")
    ap.add_argument("--keep8", default="0,4,8,16,24,40", help="K sensitive tensors kept at 8-bit")
    ap.add_argument("--full-limit", type=int, default=0, help="cap final-eval images (0=all)")
    ap.add_argument("--outdir", default="mixed_precision")
    args = ap.parse_args()

    core = ov.Core()
    out = Path(args.outdir); (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    base_buf = Path(args.ir).with_suffix(".bin").read_bytes()
    weights = list_weights(args.ir)
    print(f"weight tensors: {len(weights)}")

    files = load_files(args.data)
    ys_full = np.array([y for _, y in (files[:args.full_limit // 2] + files[-args.full_limit // 2:]
                                       if args.full_limit else files)])
    full = files[:args.full_limit // 2] + files[-args.full_limit // 2:] if args.full_limit else files

    # balanced sensitivity subset
    g = [f for f in files if f[1] == 0][:args.sens_n // 2]
    b = [f for f in files if f[1] == 1][:args.sens_n // 2]
    sub = g + b; ys_sub = np.array([y for _, y in sub])
    tmp_xml = out / "models" / "_tmp.xml"; shutil.copy(args.ir, tmp_xml)

    # FP32 reference logits on subset
    _, ref_logit = eval_model(core, args.ir, sub, args.size)

    # ---- 1) sensitivity: each tensor alone -> 4-bit ----
    print("== sensitivity (each tensor @4-bit, logit MSE vs FP32) ==")
    sens = []
    for idx, name, shape, off, size in weights:
        write_mixed(base_buf, weights, {idx: 4}, tmp_xml.with_suffix(".bin"))
        _, lg = eval_model(core, tmp_xml, sub, args.size)
        mse = float(np.mean((lg - ref_logit) ** 2))
        acc = confusion(ys_sub, lg, 0.0)["accuracy"]
        sens.append(dict(idx=idx, name=name, nel=int(np.prod(shape)), mse=mse, acc4=acc))
    sens.sort(key=lambda s: -s["mse"])           # most sensitive first
    print("  top-8 most sensitive tensors:")
    for s in sens[:8]:
        print(f"    #{s['idx']:2d} nel={s['nel']:>8d}  logitMSE {s['mse']:>9.3f}  acc@4 {s['acc4']:.3f}  {s['name']}")
    order8 = [s["idx"] for s in sens]            # priority order for 8-bit

    # ---- 2)+3) mixed configs: K most-sensitive at 8-bit, rest at 4-bit ----
    print("\n== mixed-precision configs (full eval) ==")
    configs = {}
    Ks = [int(x) for x in args.keep8.split(",")]
    for K in Ks:
        assign = {i: 4 for i, *_ in weights}
        for i in order8[:K]:
            assign[i] = 8
        xmlK = out / "models" / f"mix_keep8_{K}.xml"; shutil.copy(args.ir, xmlK)
        write_mixed(base_buf, weights, assign, xmlK.with_suffix(".bin"))
        ps, ls = eval_model(core, xmlK, full, args.size)
        c = confusion(ys_full, ps, 0.5)
        c["auc"] = round(safe_auc(ys_full, ls), 4)
        # accuracy if we re-tune the logit threshold (quantization shifts the operating point)
        grid = np.linspace(np.nanmin(ls), np.nanmax(ls), 201)
        c["acc_best_logit"] = round(max(float(((ls >= t).astype(int) == ys_full).mean()) for t in grid), 4)
        c["keep8"] = K; c["avg_bits"] = round(avg_bits(weights, assign), 3)
        configs[K] = c
        print(f"  keep8={K:3d}  avg_bits {c['avg_bits']:.2f}  acc@.5 {c['accuracy']:.4f}  "
              f"acc_bestlogit {c['acc_best_logit']:.4f}  AUC {c['auc']:.4f}  "
              f"(TN{c['TN']} FP{c['FP']} FN{c['FN']} TP{c['TP']})")

    # ---- pick the knee: smallest avg_bits whose best-logit accuracy >= 0.97 ----
    ok = [K for K in Ks if configs[K]["acc_best_logit"] >= 0.97]
    chosen = min(ok, key=lambda K: configs[K]["avg_bits"]) if ok else max(Ks, key=lambda K: configs[K]["acc_best_logit"])
    print(f"\nchosen operating point: keep8={chosen}  "
          f"avg_bits {configs[chosen]['avg_bits']:.2f}  acc {configs[chosen]['acc_best_logit']:.4f}")

    # ---- figures ----
    Kx = sorted(configs)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot([configs[k]["avg_bits"] for k in Kx], [configs[k]["accuracy"] for k in Kx], "o-", label="acc @0.5")
    ax.plot([configs[k]["avg_bits"] for k in Kx], [configs[k]["acc_best_logit"] for k in Kx], "s--", label="acc @best logit thr")
    ax.axhline(0.977, color="g", ls=":", lw=1, label="FP32 (0.977)")
    for k in Kx:
        ax.annotate(f"K={k}", (configs[k]["avg_bits"], configs[k]["acc_best_logit"]),
                    textcoords="offset points", xytext=(0, 7), fontsize=8, ha="center")
    ax.set_xlabel("average weight bits/param (memory)"); ax.set_ylabel("accuracy")
    ax.set_title("Mixed-precision weights: accuracy vs weight memory\n(K most-sensitive tensors kept 8-bit, rest 4-bit)")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(out / "figures" / "pareto_accuracy_vs_bits.png", dpi=130); plt.close(fig)

    c = configs[chosen]
    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    cm = np.array([[c["TN"], c["FP"]], [c["FN"], c["TP"]]])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred good", "pred bad"], fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["good", "bad"], fontsize=8)
    ax.set_title(f"Mixed 8/4-bit  (keep8={chosen}, {c['avg_bits']:.2f} bits/param)\n"
                 f"acc {c['accuracy']:.3f} @0.5, {c['acc_best_logit']:.3f} @best  AUC {c['auc']:.3f}", fontsize=8.5)
    fig.tight_layout(); fig.savefig(out / "figures" / "confusion_mixed.png", dpi=130); plt.close(fig)

    (out / "mixed_precision.json").write_text(json.dumps(
        {"sensitivity": sens, "configs": configs, "chosen_keep8": chosen,
         "n_test": len(full), "n_weights": len(weights)}, indent=2))
    for f in out.glob("models/_tmp.*"):
        f.unlink()
    print(f"\nfigures -> {out/'figures'}/   json -> {out/'mixed_precision.json'}")


if __name__ == "__main__":
    main()
