"""
Host-side YOLOv3 decode + NMS (numpy only — no TensorFlow).

For the FPGA deployment, the DLA runs only the convolutional graph and emits the 3 raw
detection-head tensors. All the YOLO "head math" (sigmoid, exp, grid offset, anchor
scaling) and NMS run here on the host CPU. This module is the reference host
post-processing; the same logic goes next to the FPGA runtime.

Each raw output has shape (1, gh, gw, num_anchors, 5 + num_classes):
    [tx, ty, tw, th, objectness, *class_logits]
"""
import os
import sys
import numpy as np
import cv2

# Anchors from scripts/anchors.json (k-means) if present, else stock COCO anchors.
# masks[i] pairs with the output whose grid is the i-th SMALLEST (largest stride first).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anchors_config import load_anchors
ANCHORS, MASKS = load_anchors()


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))   # clip avoids exp overflow


def _decode_one(raw, anchors_sub):
    """raw: (gh, gw, A, 5+C) -> boxes (N,4) normalized x1y1x2y2, scores (N,C)."""
    gh, gw, A, _ = raw.shape
    xy = _sigmoid(raw[..., 0:2])
    wh = raw[..., 2:4]
    obj = _sigmoid(raw[..., 4:5])
    cls = _sigmoid(raw[..., 5:])
    gx, gy = np.meshgrid(np.arange(gw), np.arange(gh))
    grid = np.stack([gx, gy], -1).reshape(gh, gw, 1, 2).astype(np.float32)
    xy = (xy + grid) / np.array([gw, gh], np.float32)
    wh = np.exp(np.clip(wh, -10, 10)) * anchors_sub.reshape(1, 1, A, 2)  # clip avoids exp overflow
    x1y1 = xy - wh / 2.0
    x2y2 = xy + wh / 2.0
    boxes = np.concatenate([x1y1, x2y2], -1).reshape(-1, 4)
    scores = (obj * cls).reshape(-1, cls.shape[-1])
    return boxes, scores


def decode_and_nms(raw_outputs, w0, h0, score_thresh=0.25, iou_thresh=0.45):
    """raw_outputs: list of np arrays, each (1, gh, gw, A, 5+C) in any scale order.
    Returns list of (x1, y1, x2, y2, score, cls) in pixel coords of a w0 x h0 image."""
    # squeeze batch; order outputs by grid size so each matches the right anchor mask
    outs = [o[0] if o.ndim == 5 else o for o in raw_outputs]
    outs = sorted(outs, key=lambda a: a.shape[0])      # smallest grid first -> MASKS[0]
    all_boxes, all_scores = [], []
    for raw, mask in zip(outs, MASKS):
        b, s = _decode_one(raw, ANCHORS[mask])
        all_boxes.append(b)
        all_scores.append(s)
    boxes = np.concatenate(all_boxes, 0)
    scores = np.concatenate(all_scores, 0)

    cls_id = scores.argmax(1)
    cls_score = scores.max(1)
    keep = cls_score >= score_thresh
    boxes, cls_score, cls_id = boxes[keep], cls_score[keep], cls_id[keep]
    if len(cls_score) == 0:
        return []
    rects = [[int(x1 * w0), int(y1 * h0), int((x2 - x1) * w0), int((y2 - y1) * h0)]
             for x1, y1, x2, y2 in boxes]
    idxs = cv2.dnn.NMSBoxes(rects, cls_score.tolist(), score_thresh, iou_thresh)
    dets = []
    for i in np.array(idxs).flatten():
        x, y, ww, hh = rects[i]
        dets.append((x, y, x + ww, y + hh, float(cls_score[i]), int(cls_id[i])))
    return dets
