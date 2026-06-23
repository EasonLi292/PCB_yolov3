#!/usr/bin/env python3
"""
Preprocess the built YOLO datasets for the neural network:
  - convert every image to grayscale (single channel)
  - letterbox-resize to a fixed SIZE x SIZE (aspect ratio preserved, padded with 114)
  - transform the YOLO labels to match the letterboxed geometry

Non-destructive: reads datasets/<name>/, writes datasets/<name>_gray<SIZE>/ with the
same train/val/test/{images,labels} structure and an updated data.yaml.

Usage:
    python scripts/preprocess_images.py [pku|deeppcb|dspcbsd|all] [--size 640]
"""
import sys, shutil
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "datasets"
SIZE = 640
PAD = 114                      # standard YOLO letterbox gray
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

SETS = {
    "pku":     "unified_pku_yolo",
    "deeppcb": "deeppcb_yolo",
    "dspcbsd": "dspcbsd_yolo",
}


def letterbox_gray(img, size):
    """Grayscale + letterbox to size x size. Returns (out_img, ratio, pad_x, pad_y)."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h0, w0 = img.shape[:2]
    r = min(size / w0, size / h0)
    nw, nh = round(w0 * r), round(h0 * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    canvas = np.full((size, size), PAD, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, r, px, py, w0, h0


def remap_labels(text, r, px, py, w0, h0, size):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        c, cx, cy, bw, bh = line.split()[:5]
        cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
        # de-normalize on original -> apply resize+pad -> re-normalize on size
        ncx = (cx * w0 * r + px) / size
        ncy = (cy * h0 * r + py) / size
        nbw = (bw * w0 * r) / size
        nbh = (bh * h0 * r) / size
        out.append(f"{c} {ncx:.6f} {ncy:.6f} {nbw:.6f} {nbh:.6f}")
    return "\n".join(out) + ("\n" if out else "")


def process(name, size):
    src = DS / name
    dst = DS / f"{name}_gray{size}"
    if dst.exists():
        shutil.rmtree(dst)
    total = 0
    for split in ("train", "val", "test"):
        img_dir = src / split / "images"
        lbl_dir = src / split / "labels"
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            out_img, r, px, py, w0, h0 = letterbox_gray(img, size)
            out_img_path = dst / split / "images" / f"{img_path.stem}.png"
            out_img_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_img_path), out_img)            # PNG: lossless grayscale
            lp = lbl_dir / f"{img_path.stem}.txt"
            txt = lp.read_text() if lp.exists() else ""
            out_lbl_path = dst / split / "labels" / f"{img_path.stem}.txt"
            out_lbl_path.parent.mkdir(parents=True, exist_ok=True)
            out_lbl_path.write_text(remap_labels(txt, r, px, py, w0, h0, size))
            total += 1
    # data.yaml: copy class section from source, fix path
    src_yaml = (src / "data.yaml").read_text().splitlines()
    new_yaml = [f"path: {dst}", "train: train/images", "val: val/images", "test: test/images", ""]
    keep = False
    for line in src_yaml:
        if line.startswith("nc:"):
            keep = True
        if keep:
            new_yaml.append(line)
    new_yaml.append(f"# preprocessed: grayscale, letterboxed to {size}x{size} (pad={PAD}), channels=1")
    (dst / "data.yaml").write_text("\n".join(new_yaml) + "\n")
    print(f"[{name}] -> {dst.name}  ({total} images, grayscale {size}x{size})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--size" in sys.argv:
        SIZE = int(sys.argv[sys.argv.index("--size") + 1])
    what = args[0] if args else "all"
    targets = list(SETS) if what == "all" else [what]
    for t in targets:
        process(SETS[t], SIZE)
    print("done.")
