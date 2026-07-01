#!/usr/bin/env python3
"""
tf.data input pipeline for the good/bad ResNet classifier.

Expects a dataset laid out as:

    <root>/<split>/good/*.png|jpg
    <root>/<split>/bad/*.png|jpg

Images are loaded as 3-channel color, resized to `size`, lightly augmented (train
only), and passed through ResNet preprocess_input. A grayscale source (e.g. the B&W
whole-board sets) decodes to 3 equal channels, so the same path handles both.
Labels: good = 0, defective = 1 (positive class = defective).
"""
import tensorflow as tf
from pathlib import Path
from resnet50_tf import preprocess_batch

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
AUTOTUNE = tf.data.AUTOTUNE


def list_split(split_dir):
    """Return (paths, labels) for one split dir containing good/ and bad/."""
    split_dir = Path(split_dir)
    paths, labels = [], []
    for cls, lab in (("good", 0), ("bad", 1)):
        d = split_dir / cls
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                paths.append(str(p))
                labels.append(lab)
    return paths, labels


def _load(path, label, size, augment):
    raw = tf.io.read_file(path)
    # decode as 3-channel color; a grayscale source decodes to 3 equal channels, so
    # this works for both the color PKU patches and the B&W DeepPCB whole-board sets.
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, (size, size))
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.10 * 255)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        img = tf.image.random_saturation(img, 0.85, 1.15)
        img = tf.clip_by_value(img, 0.0, 255.0)
    img = preprocess_batch(img)
    return img, tf.cast(label, tf.float32)


def make_dataset(split_dir, size=224, batch=32, shuffle=False, augment=False):
    """Build a batched tf.data.Dataset and return (ds, n_images, class_counts)."""
    paths, labels = list_split(split_dir)
    n = len(paths)
    counts = {"good": labels.count(0), "bad": labels.count(1)}
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(min(n, 4096), reshuffle_each_iteration=True)
    ds = ds.map(lambda p, l: _load(p, l, size, augment), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch).prefetch(AUTOTUNE)
    return ds, n, counts


def class_weights(counts):
    """Inverse-frequency weights so the rarer class isn't ignored under imbalance."""
    total = counts["good"] + counts["bad"]
    out = {}
    for cls, lab in (("good", 0), ("bad", 1)):
        c = counts[cls]
        out[lab] = (total / (2.0 * c)) if c else 0.0
    return out
