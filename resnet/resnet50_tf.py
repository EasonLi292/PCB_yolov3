#!/usr/bin/env python3
"""
ResNet-50 binary PCB classifier (good board vs. defective board).

This is the classification counterpart to the YOLOv3 detector in ../yolov3/.
Instead of localizing every defect, it makes one board-level call:

    output = P(defective)   in [0, 1]      (label: good = 0, defective = 1)

Design choices (and why):
  * ResNet-50 ImageNet backbone, kept as a NESTED model named "resnet50" so the
    whole backbone can be frozen/unfrozen with one `.trainable =` toggle, exactly
    like the YOLO transfer-learning phases.
  * Single sigmoid logit (not 2-class softmax): standard for binary, and the
    threshold is then tunable at eval time to trade precision vs. recall.
  * Positive class = DEFECTIVE. We care most about *catching bad boards*, so recall
    on the positive class is the headline safety metric.
  * COLOR input: most boards (the PKU sources we train on) are color photos, and color
    is the native input for the ImageNet ResNet backbone, so the pipeline keeps RGB.
    (The detector stays grayscale; this classifier does not.) The FPGA IR input is
    [1, size, size, 3] either way, so color is essentially free.

ResNet-50 is a first-class reference model for the Intel FPGA AI Suite DLA (plain
conv / BN / ReLU / add / global-pool / dense), so this graph maps cleanly to the
Agilex 7 target — no custom ops, unlike the YOLO decode/NMS.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input

BACKBONE_NAME = "resnet50"


def build_resnet50(size=224, dropout=0.3, weights="imagenet", freeze_backbone=True,
                   n_classes=1):
    """Build the ResNet-50 PCB classifier.

    Args:
        size:  square input resolution (default 224 = ImageNet native).
        dropout: dropout before the classifier head.
        weights: "imagenet" for transfer learning, or None for scratch.
        freeze_backbone: True => only the head trains (transfer phase 1).
        n_classes: 1 => binary good/defective (sigmoid, P(defective)); the default and
            what the FPGA export expects. >1 => multi-class defect-TYPE head (softmax),
            used by train_multiclass.py / eval_multiclass.py.
    """
    inp = layers.Input((size, size, 3), name="input")
    base = ResNet50(include_top=False, weights=weights, pooling="avg")
    base._name = BACKBONE_NAME
    base.trainable = not freeze_backbone
    x = base(inp)                                    # -> (None, 2048)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    if n_classes == 1:
        out = layers.Dense(1, activation="sigmoid", name="defect_prob")(x)
    else:
        out = layers.Dense(n_classes, activation="softmax", name="defect_type")(x)
    return Model(inp, out, name="resnet50_pcb")


def set_backbone_trainable(model, trainable):
    """Freeze (False) or unfreeze (True) the entire ResNet-50 backbone in one call."""
    model.get_layer(BACKBONE_NAME).trainable = trainable


def preprocess_batch(x):
    """Apply the canonical ResNet preprocessing (expects 0-255 RGB-order floats)."""
    return preprocess_input(x)
