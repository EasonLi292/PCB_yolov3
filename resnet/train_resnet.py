#!/usr/bin/env python3
"""
Train the ResNet-50 good/bad PCB classifier via transfer learning.

Two-phase recipe (same idea as the YOLO detector):
  Phase 1 (default): freeze the ImageNet backbone, train only the new head.
  Phase 2 (--unfreeze, low --lr): unfreeze the backbone to adapt features to the
           grayscale-PCB domain.

Class imbalance (we expect many more GOOD than BAD, or vice-versa) is handled with
inverse-frequency class weights, so the rarer class is not ignored.

Examples:
  python resnet/train_resnet.py --data datasets/pcb_goodbad --epochs 20 --batch 32
  python resnet/train_resnet.py --data datasets/pcb_goodbad --resume runs_resnet/pcb_goodbad/best.weights.h5 \
         --unfreeze --lr 1e-5 --epochs 15
"""
import argparse, os, sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tensorflow as tf
from resnet50_tf import build_resnet50, set_backbone_trainable
from data import make_dataset, class_weights

ROOT = Path(__file__).resolve().parent.parent


def save_manifest(path, info):
    """Write a JSON run manifest (config + provenance) next to the weights, so they can be
    re-tested later without re-training and the exact dataset used stays recorded."""
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
    ap.add_argument("--data", required=True, help="dataset dir with train/ val/ (good/ bad/)")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--unfreeze", action="store_true",
                    help="unfreeze the backbone (full fine-tune; use a small --lr)")
    ap.add_argument("--resume", default="", help="load these .h5 weights before training")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "runs_resnet"))
    ap.add_argument("--smoke", action="store_true", help="one train step to validate the pipeline")
    args = ap.parse_args()

    data_dir = Path(args.data)
    model = build_resnet50(size=args.size, dropout=args.dropout,
                           freeze_backbone=not args.unfreeze)
    if args.resume and Path(args.resume).exists():
        model.load_weights(args.resume)
        print(f"Resumed weights from {args.resume}")
    if args.unfreeze:
        set_backbone_trainable(model, True)
        print("Backbone UNFROZEN (full fine-tune).")
    else:
        print("Backbone FROZEN (training head only).")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")])

    train_ds, n_tr, c_tr = make_dataset(data_dir / "train", args.size, args.batch,
                                        shuffle=True, augment=not args.no_augment)
    val_ds, n_va, c_va = make_dataset(data_dir / "val", args.size, args.batch)
    cw = class_weights(c_tr)
    print(f"train: {n_tr} ({c_tr})   val: {n_va} ({c_va})")
    print(f"class weights (0=good,1=bad): {cw}")
    print(f"augmentation: {'OFF' if args.no_augment else 'ON'}")

    if args.smoke:
        xb, yb = next(iter(train_ds.take(1)))
        h = model.train_on_batch(xb, yb, return_dict=True)
        print("SMOKE OK:", {k: round(float(v), 3) for k, v in h.items()})
        return

    out = Path(args.out) / data_dir.name
    out.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", patience=3, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8,
                                         verbose=1, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(out / "best.weights.h5"), monitor="val_auc",
                                           mode="max", save_best_only=True,
                                           save_weights_only=True, verbose=1),
    ]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                     class_weight=cw, callbacks=callbacks)

    model.save_weights(str(out / "best.weights.h5"))   # guarantee the (restored-best) weights exist
    model.export(str(out / "saved_model"))     # inference SavedModel (for OpenVINO export)
    (out / "classes.txt").write_text("good\nbad\n")

    def _best(metric, mode=max):
        v = hist.history.get(metric)
        return float(mode(v)) if v else None
    save_manifest(out / "run_manifest.json", {
        "task": "binary_goodbad", "model": "resnet50",
        "weights": "best.weights.h5", "saved_model": "saved_model", "classes": ["good", "bad"],
        "dataset": str(Path(args.data).resolve()), "dataset_name": data_dir.name,
        "size": args.size, "batch": args.batch, "epochs": args.epochs, "lr": args.lr,
        "dropout": args.dropout,
        "phase": "unfrozen (full fine-tune)" if args.unfreeze else "frozen (head only)",
        "resumed_from": args.resume or None, "augment": not args.no_augment,
        "train_counts": c_tr, "val_counts": c_va,
        "best_val_auc": _best("val_auc"), "best_val_acc": _best("val_acc"),
        "note": "re-test with: python resnet/eval_resnet.py --weights <this>/best.weights.h5 "
                f"--data {args.data} --size {args.size}",
    })
    print(f"Saved best weights + inference SavedModel + run_manifest.json under {out}")


if __name__ == "__main__":
    main()
