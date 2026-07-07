#!/usr/bin/env python3
"""
tf.data input pipeline for the multi-class defect-TYPE ResNet (Goal 2).

Expects a dataset laid out with one folder per defect class:

    <root>/<split>/<class_name>/*.png|jpg

produced by mine_defect_types.py. Integer labels are assigned from the sorted union of
class folder names (stable across splits). Same preprocessing/augmentation as data.py.
"""
import tensorflow as tf
from pathlib import Path
from resnet50_tf import preprocess_batch

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
AUTOTUNE = tf.data.AUTOTUNE


def class_names(root):
    """Sorted union of class-subfolder names across all splits (stable label order)."""
    root = Path(root)
    names = set()
    for sp in ("train", "val", "test"):
        d = root / sp
        if d.is_dir():
            names.update(c.name for c in d.iterdir() if c.is_dir())
    return sorted(names)


def list_split(split_dir, names):
    split_dir = Path(split_dir)
    paths, labels = [], []
    for i, c in enumerate(names):
        d = split_dir / c
        if d.is_dir():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    paths.append(str(p)); labels.append(i)
    return paths, labels


def _load(path, label, size, augment):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, (size, size))
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.10 * 255)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        img = tf.clip_by_value(img, 0.0, 255.0)
    return preprocess_batch(img), tf.cast(label, tf.int32)


def make_dataset(split_dir, names, size=256, batch=32, shuffle=False, augment=False):
    """Return (ds, n_images, class_counts{name:count})."""
    paths, labels = list_split(split_dir, names)
    n = len(paths)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(min(max(n, 1), 4096), reshuffle_each_iteration=True)
    ds = ds.map(lambda p, l: _load(p, l, size, augment), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch).prefetch(AUTOTUNE)
    counts = {names[i]: labels.count(i) for i in range(len(names))}
    return ds, n, counts


def class_weights(counts, names):
    """Inverse-frequency weights keyed by class index (rarer defect types not ignored)."""
    total = sum(counts.values()); k = len(names); out = {}
    for i, c in enumerate(names):
        n = counts.get(c, 0)
        out[i] = (total / (k * n)) if n else 0.0
    return out
