#!/usr/bin/env python3
"""
Export a trained YOLOv3 to an FPGA-friendly OpenVINO IR for Intel FPGA AI Suite
(target: Agilex 7 F-series SoC + DLA IP).

Key difference from export_openvino.py: this emits the **raw convolutional detection
heads only** (3 outputs, NO decode / NMS in the graph). That graph is pure
conv / BN / LeakyReLU / upsample / concat / add — the operators the DLA maps cleanly.
All YOLO decode + NMS run on the host (see scripts/yolo_postprocess.py).

Produces:
  <out>/yolov3_fpga_fp32.xml/.bin           FP32 raw-output IR (feed to AI Suite)
  <out>/yolov3_fpga_int8.xml/.bin           INT8 IR (optional, --int8; needs nncf)

Reads the trained weights straight from the .h5 checkpoint, so it works even if a
training run was interrupted (the checkpoint is saved every improving epoch).

Example:
  python scripts/export_fpga.py \
      --weights runs/unified_pku_yolo_gray640/yolov3_best.weights.h5 \
      --out runs/unified_pku_yolo_gray640/openvino_fpga \
      --int8 --calib-data datasets/unified_pku_yolo_gray640/train --calib-n 200
"""
import argparse
import shutil
from pathlib import Path
import numpy as np


def build_savedmodel(weights, nc, size, sm_dir):
    """TF: build the raw 3-output graph, load weights, export a SavedModel."""
    import os, sys
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from yolov3_tf import YoloV3
    model = YoloV3(size=size, classes=nc, training=True)   # raw conv heads, no decode
    model.load_weights(weights)
    if Path(sm_dir).exists():
        shutil.rmtree(sm_dir)
    model.export(str(sm_dir))


def build_savedmodel_subprocess(weights, nc, size, sm_dir):
    """Run the TF SavedModel build in a child process.

    TensorFlow and OpenVINO crash if imported in the same interpreter (libc++ mutex
    abort on macOS); isolating TF in a subprocess keeps the two apart and is portable.
    """
    import subprocess, sys
    cmd = [sys.executable, __file__, "--_build_sm",
           "--weights", str(weights), "--nc", str(nc),
           "--size", str(size), "--sm-dir", str(sm_dir)]
    subprocess.run(cmd, check=True)


def convert_fp32(sm_dir, size, xml_out):
    import openvino as ov
    ov_model = ov.convert_model(str(sm_dir), input=[[1, size, size, 3]])
    ov.save_model(ov_model, str(xml_out))
    return xml_out


def _calib_preprocess(path, size):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)          # black & white
    img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)               # tile to 3ch
    img3 = cv2.resize(img3, (size, size)).astype(np.float32) / 255.0
    return img3[None, ...]                                     # (1,H,W,3) NHWC


def _head_conv_names(model):
    """Find the 3 final detection-head convolutions (the ones feeding each output).
    These produce wide-range YOLO logits (tx,ty,tw,th,obj,cls) that don't survive INT8,
    so we keep them in FP32 to avoid the classic YOLO-quantization accuracy collapse."""
    names = []
    for out in model.outputs:
        seen, stack = set(), [out.get_node()]
        while stack:
            n = stack.pop()
            if n.get_name() in seen:
                continue
            seen.add(n.get_name())
            if "Convolution" in n.get_type_name():
                names.append(n.get_friendly_name())
                break
            for inp in n.inputs():
                stack.append(inp.get_source_output().get_node())
    return names


def quantize_int8(fp32_xml, calib_dir, size, xml_out, n):
    import openvino as ov
    import nncf
    core = ov.Core()
    model = core.read_model(str(fp32_xml))
    imgs = sorted(p for p in (Path(calib_dir) / "images").iterdir()
                  if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})[: n * 3]
    if not imgs:
        raise SystemExit(f"no calibration images under {calib_dir}/images")

    ignored = _head_conv_names(model)
    print("  keeping detection heads in FP32:", ignored)

    def transform_fn(path):
        return _calib_preprocess(path, size)

    calibration = nncf.Dataset(imgs, transform_fn)
    quantized = nncf.quantize(
        model, calibration, subset_size=min(n, len(imgs)),
        ignored_scope=nncf.IgnoredScope(names=ignored))
    ov.save_model(quantized, str(xml_out))
    return xml_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="trained .h5 checkpoint")
    ap.add_argument("--out", help="output dir for the FPGA IR")
    ap.add_argument("--nc", type=int, default=6, help="number of classes")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--int8", action="store_true", help="also emit an INT8 IR (needs nncf)")
    ap.add_argument("--calib-data", help="dataset split dir for INT8 calibration (has images/)")
    ap.add_argument("--calib-n", type=int, default=200)
    ap.add_argument("--classes", help="optional classes.txt to copy next to the IR")
    # internal: build the SavedModel only (runs in a TF-only child process)
    ap.add_argument("--_build_sm", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--sm-dir", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._build_sm:                       # child process: TF only, no OpenVINO
        build_savedmodel(args.weights, args.nc, args.size, args.sm_dir)
        return

    if not args.out:
        ap.error("--out is required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sm_dir = out / "_saved_model_raw"

    print("[1/2] Building raw-output SavedModel from checkpoint (TF subprocess)...")
    build_savedmodel_subprocess(args.weights, args.nc, args.size, sm_dir)

    print("[2/2] Converting to FP32 OpenVINO IR (raw conv heads, decode on host)...")
    fp32 = convert_fp32(sm_dir, args.size, out / "yolov3_fpga_fp32.xml")
    print("  ->", fp32)

    if args.int8:
        calib = args.calib_data or args.weights  # require explicit calib dir
        if not args.calib_data:
            raise SystemExit("--int8 requires --calib-data <dataset split dir>")
        print(f"[INT8] Post-training quantization with NNCF ({args.calib_n} images)...")
        int8 = quantize_int8(fp32, args.calib_data, args.size,
                             out / "yolov3_fpga_int8.xml", args.calib_n)
        print("  ->", int8)

    if args.classes and Path(args.classes).exists():
        shutil.copy(args.classes, out / "classes.txt")
    shutil.rmtree(sm_dir, ignore_errors=True)   # SavedModel was just an intermediate
    print("Done. FPGA IR written to", out)


if __name__ == "__main__":
    main()
