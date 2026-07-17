#!/usr/bin/env python3
"""Preprocess JPEGs -> raw float32 NHWC .bin files for dla_benchmark.

dla_benchmark misreads this model's NHWC input [1,512,512,3] as NCHW (C=512), so it won't
ingest JPEGs and fills with random data. Feeding raw .bin sidesteps that AND lets us bake in
the EXACT caffe preprocessing (INTER_AREA resize + BGR + ImageNet mean) so the emulator output
is directly comparable to the CPU reference (release_resnet_512/emu/ref_cpu.npz).

Each .bin is the model input tensor in NHWC memory order, float32:  512*512*3*4 = 3,145,728 bytes.
When you run dla_benchmark on these, DROP -bgr and -mean_values (already applied here).

Usage:
  python make_bins.py <src_image_dir> <dst_bin_dir> [limit]
  python make_bins.py .../test_dataset/bad ./bad_bins 200
"""
import glob, sys
from pathlib import Path
import numpy as np
import cv2

MEAN = np.array([103.939, 116.779, 123.68], np.float32)  # keras caffe BGR means


def preprocess(path, size=512):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"unreadable: {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = img[..., ::-1] - MEAN                      # RGB->BGR, subtract means
    return np.ascontiguousarray(img, dtype=np.float32)   # [512,512,3] NHWC


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    Path(dst).mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in glob.glob(f"{src}/*") if Path(p).is_file())
    if limit:
        files = files[:limit]
    n = 0
    for i, f in enumerate(files):
        arr = preprocess(f)
        arr.tofile(f"{dst}/{i:05d}_{Path(f).stem}.bin")
        n += 1
    print(f"wrote {n} bins to {dst}  ({arr.nbytes} bytes each, NHWC float32)")
    # manifest so you can map bin order -> source image later
    Path(f"{dst}/manifest.txt").write_text(
        "\n".join(f"{i:05d}\t{Path(f).name}" for i, f in enumerate(files)))


if __name__ == "__main__":
    main()
