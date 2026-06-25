#!/usr/bin/env python3
"""
Train YOLOv3 (TensorFlow/Keras) on a PCB-defect YOLO dataset via transfer learning.

Pipeline:
  1. Build YOLOv3 with our class count (nc from data.yaml).
  2. Load the official pretrained Darknet weights (yolov3.weights, COCO/80-class).
  3. Transfer the Darknet-53 backbone + the class-independent FPN necks; the three
     detection heads (yolo_output_*) are REDEFINED for our nc classes and trained
     from scratch -- i.e. "redefine the last few layers for our use case".
  4. Freeze the backbone and train the necks + new heads (optionally unfreeze later).

Input images may be single-channel grayscale (the *_gray640 sets); the data pipeline
tiles them to 3 channels so the pretrained RGB backbone stays valid.

Examples:
  python scripts/train_yolov3.py --data datasets/unified_pku_yolo_gray640 \
         --weights weights/yolov3.weights --epochs 50 --batch 8
  python scripts/train_yolov3.py --data datasets/unified_pku_yolo_gray640 --unfreeze ...
"""
import argparse, os, sys, urllib.request
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yolov3_tf import (
    YoloV3, YoloLoss, yolo_anchors, yolo_anchor_masks,
    load_darknet_weights, freeze_all, make_dataset,
)

WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
ROOT = Path(__file__).resolve().parent.parent


def read_yaml_names(data_dir: Path):
    """Minimal data.yaml reader: returns (nc, names)."""
    names, nc = [], None
    for line in (data_dir / "data.yaml").read_text().splitlines():
        s = line.strip()
        if s.startswith("nc:"):
            nc = int(s.split(":")[1].strip())
        elif ":" in s and s.split(":")[0].strip().isdigit():
            names.append(s.split(":", 1)[1].strip())
    if nc is None:
        nc = len(names)
    return nc, names


def maybe_download_weights(path: Path):
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pretrained yolov3.weights (237 MB) -> {path} ...")
    urllib.request.urlretrieve(WEIGHTS_URL, path)
    return path


def build_transfer_model(size, nc, weights_path):
    """Build training model and transfer pretrained backbone + necks."""
    model = YoloV3(size=size, classes=nc, training=True)

    if weights_path and Path(weights_path).exists():
        print("Loading pretrained Darknet weights into an 80-class reference model...")
        ref = YoloV3(size=size, classes=80, training=True)
        load_darknet_weights(ref, str(weights_path))
        # Backbone + necks are class-independent -> transfer; heads stay fresh.
        for layer_name in ["yolo_darknet", "yolo_conv_0", "yolo_conv_1", "yolo_conv_2"]:
            model.get_layer(layer_name).set_weights(
                ref.get_layer(layer_name).get_weights())
        print("Transferred:", "darknet backbone + 3 FPN necks; "
              "heads (yolo_output_*) reinitialized for", nc, "classes.")
    else:
        print("WARNING: no pretrained weights found -> training from scratch.")

    # Freeze the backbone for transfer-learning phase.
    freeze_all(model.get_layer("yolo_darknet"), frozen=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset dir (has train/, val/, data.yaml)")
    ap.add_argument("--weights", default=str(ROOT / "weights" / "yolov3.weights"))
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--unfreeze", action="store_true",
                    help="also train the backbone (full fine-tune, use a small lr)")
    ap.add_argument("--resume", default="",
                    help="load these .h5 weights before training (e.g. phase-1 best)")
    ap.add_argument("--no-augment", action="store_true",
                    help="disable online augmentation (flips, 90° rotation, brightness)")
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--steps", type=int, default=0,
                    help="limit steps_per_epoch (0 = full epoch); useful for a quick run")
    ap.add_argument("--download-weights", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="build + 1 train step on a tiny batch to validate the pipeline")
    args = ap.parse_args()

    data_dir = Path(args.data)
    nc, names = read_yaml_names(data_dir)
    print(f"Dataset: {data_dir.name}  classes={nc}  names={names}")

    wpath = Path(args.weights)
    if args.download_weights:
        wpath = maybe_download_weights(wpath)

    if args.resume and Path(args.resume).exists():
        # Continuing from our own trained weights -> skip the COCO transfer entirely
        # (it would just be overwritten). Build a plain model and load our checkpoint.
        model = YoloV3(size=args.size, classes=nc, training=True)
        model.load_weights(args.resume)
        print(f"Resumed weights from {args.resume} (skipped COCO transfer)")
        freeze_all(model.get_layer("yolo_darknet"), frozen=True)  # default; --unfreeze flips it
    else:
        # Fresh transfer learning: COCO backbone + necks, new heads.
        model = build_transfer_model(args.size, nc, wpath)

    if args.unfreeze:
        freeze_all(model.get_layer("yolo_darknet"), frozen=False)
        print("Backbone UNFROZEN (full fine-tune).")

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss = [YoloLoss(yolo_anchors[mask], classes=nc) for mask in yolo_anchor_masks]
    model.compile(optimizer=optimizer, loss=loss)

    train_ds, n_train = make_dataset(
        data_dir / "train", yolo_anchors, yolo_anchor_masks, args.size, nc,
        args.batch, shuffle=True, augment_data=not args.no_augment)
    val_ds, n_val = make_dataset(
        data_dir / "val", yolo_anchors, yolo_anchor_masks, args.size, nc,
        args.batch, shuffle=False)
    print(f"online augmentation: {'OFF' if args.no_augment else 'ON (flips, 90° rot, brightness/contrast)'}")
    print(f"train images: {n_train}  val images: {n_val}")

    if args.smoke:
        print("SMOKE TEST: one train step on a single batch...")
        xb, yb = next(iter(train_ds.take(1)))
        h = model.train_on_batch(xb, yb, return_dict=True)
        print("  forward+backward OK. loss:", {k: round(float(v), 3) for k, v in h.items()})
        return

    out = Path(args.out) / data_dir.name
    out.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=8, verbose=1, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            str(out / "yolov3_best.weights.h5"), verbose=1,
            save_best_only=True, save_weights_only=True),
    ]
    try:                       # TensorBoard logging is optional
        import tensorboard  # noqa: F401
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=str(out / "logs")))
    except ImportError:
        print("(tensorboard not installed -> skipping TensorBoard logs)")
    fit_kw = {}
    if args.steps:
        fit_kw["steps_per_epoch"] = args.steps
        fit_kw["validation_steps"] = max(1, args.steps // 4)
    model.fit(train_ds, epochs=args.epochs, validation_data=val_ds,
              callbacks=callbacks, **fit_kw)

    # Save an inference model with decoding but NMS-free (OpenVINO-convertible;
    # NMS is done in export_openvino.py postprocessing).
    infer = YoloV3(size=args.size, classes=nc, training=False, nms=False)
    infer.set_weights(model.get_weights())
    infer.export(str(out / "saved_model"))      # TF SavedModel
    (out / "classes.txt").write_text("\n".join(names) + "\n")
    print(f"Saved training weights + inference SavedModel under {out}")


if __name__ == "__main__":
    main()
