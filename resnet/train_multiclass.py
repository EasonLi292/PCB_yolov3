#!/usr/bin/env python3
"""
Train the multi-class ResNet-50 defect-TYPE classifier (Goal 2).

Same two-phase transfer recipe as train_resnet.py, but the head is softmax over the 6
defect classes (built via build_resnet50(n_classes=K)) and the loss is categorical.
Class imbalance across defect types is handled with inverse-frequency class weights.

Examples:
  python resnet/train_multiclass.py --data datasets/pcb_defect_types --size 256 --epochs 20
  python resnet/train_multiclass.py --data datasets/pcb_defect_types --size 256 \
         --resume runs_resnet/pcb_defect_types/best.weights.h5 --unfreeze --lr 1e-5 --epochs 15
"""
import argparse, os, sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tensorflow as tf
from resnet50_tf import build_resnet50, set_backbone_trainable
import data_multiclass as dm

ROOT = Path(__file__).resolve().parent.parent


def save_manifest(path, info):
    """JSON run manifest (config + provenance) next to the weights, for later re-testing."""
    import json, subprocess, datetime
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT)).decode().strip()
    except Exception:
        info["git_commit"] = "unknown"
    info["created"] = datetime.datetime.now().isoformat(timespec="seconds")
    Path(path).write_text(json.dumps(info, indent=2))
    print(f"wrote manifest -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with train/ val/ <class>/ folders")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--unfreeze", action="store_true", help="full fine-tune (use small --lr)")
    ap.add_argument("--resume", default="")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "runs_resnet"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data)
    names = dm.class_names(data_dir)
    K = len(names)
    print(f"{K} classes: {names}")

    model = build_resnet50(size=args.size, dropout=args.dropout,
                           freeze_backbone=not args.unfreeze, n_classes=K)
    if args.resume and Path(args.resume).exists():
        model.load_weights(args.resume); print(f"Resumed {args.resume}")
    if args.unfreeze:
        set_backbone_trainable(model, True); print("Backbone UNFROZEN.")
    else:
        print("Backbone FROZEN (head only).")

    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])

    tr, n_tr, c_tr = dm.make_dataset(data_dir / "train", names, args.size, args.batch,
                                     shuffle=True, augment=not args.no_augment)
    va, n_va, c_va = dm.make_dataset(data_dir / "val", names, args.size, args.batch)
    cw = dm.class_weights(c_tr, names)
    print(f"train: {n_tr} {c_tr}\nval:   {n_va} {c_va}\nclass weights: {cw}")

    if args.smoke:
        xb, yb = next(iter(tr.take(1)))
        h = model.train_on_batch(xb, yb, return_dict=True)
        print("SMOKE OK:", {k: round(float(v), 3) for k, v in h.items()}); return

    out = Path(args.out) / data_dir.name
    out.mkdir(parents=True, exist_ok=True)
    cbs = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=8,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(str(out / "best.weights.h5"), monitor="val_acc",
                                           mode="max", save_best_only=True,
                                           save_weights_only=True, verbose=1),
    ]
    hist = model.fit(tr, validation_data=va, epochs=args.epochs, class_weight=cw, callbacks=cbs)
    model.save_weights(str(out / "best.weights.h5"))   # guarantee the (restored-best) weights exist
    model.export(str(out / "saved_model"))
    (out / "classes.txt").write_text("\n".join(names) + "\n")
    save_manifest(out / "run_manifest.json", {
        "task": "defect_type_multiclass", "model": "resnet50", "n_classes": K,
        "weights": "best.weights.h5", "saved_model": "saved_model", "classes": names,
        "dataset": str(Path(args.data).resolve()), "dataset_name": data_dir.name,
        "size": args.size, "batch": args.batch, "epochs": args.epochs, "lr": args.lr,
        "dropout": args.dropout,
        "phase": "unfrozen (full fine-tune)" if args.unfreeze else "frozen (head only)",
        "resumed_from": args.resume or None, "augment": not args.no_augment,
        "train_counts": c_tr, "val_counts": c_va,
        "best_val_acc": float(max(hist.history.get("val_acc", [0]) or [0])),
        "note": "re-test with: python resnet/eval_multiclass.py --weights <this>/best.weights.h5 "
                f"--data {args.data} --size {args.size}",
    })
    print(f"Saved best weights + SavedModel + classes.txt + run_manifest.json under {out}")


if __name__ == "__main__":
    main()
