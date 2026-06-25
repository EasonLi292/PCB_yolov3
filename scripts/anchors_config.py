"""Single source of truth for YOLOv3 anchors, shared by training (yolov3_tf.py) and
host-side decode (yolo_postprocess.py) so they can never drift apart.

If scripts/anchors.json exists (written by compute_anchors.py), those anchors are used;
otherwise the stock COCO YOLOv3 anchors. Anchors are fractions of the image side, sorted
ascending by area; masks pair the largest 3 with the coarsest grid.
"""
import json
from pathlib import Path
import numpy as np

_STOCK = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                   (59, 119), (116, 90), (156, 198), (373, 326)], np.float32) / 416.0
_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]


def load_anchors():
    """Return (anchors float32 (9,2), masks list-of-lists)."""
    f = Path(__file__).with_name("anchors.json")
    if f.exists():
        d = json.loads(f.read_text())
        return np.array(d["anchors"], np.float32), [list(m) for m in d["masks"]]
    return _STOCK.copy(), [list(m) for m in _MASKS]
