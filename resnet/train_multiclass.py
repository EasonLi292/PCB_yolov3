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
    model.fit(tr, validation_data=va, epochs=args.epochs, class_weight=cw, callbacks=cbs)
    model.export(str(out / "saved_model"))
    (out / "classes.txt").write_text("\n".join(names) + "\n")
    print(f"Saved best weights + SavedModel + classes.txt under {out}")


if __name__ == "__main__":
    main()
