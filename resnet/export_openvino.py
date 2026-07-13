#!/usr/bin/env python3
"""
Export the trained ResNet-50 good/bad classifier to OpenVINO IR for the Intel FPGA
AI Suite (Agilex 7 F-series).

ResNet-50 is a reference model for the AI Suite DLA -- the whole graph (conv / BN /
ReLU / add / global-avg-pool / dense / sigmoid) maps to hardware with no custom ops,
so unlike the detector there is no raw-head / host-decode split: one clean IR with a
single [1, size, size, 3] input and a single [1, 1] P(defective) output.

Build pattern mirrors yolov3/export_fpga.py:
  1. Build the model + load weights + Keras `model.export()` to a SavedModel, run in a
     SEPARATE process (TensorFlow and OpenVINO abort if imported in one interpreter).
  2. Convert SavedModel -> IR with `ov.convert_model(..., input=[[1,size,size,3]])`.

Usage:
  python resnet/export_openvino.py --weights runs_resnet/pcb_goodbad/best.weights.h5 \
         --out release_resnet
"""
import argparse, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build_savedmodel(weights, size, sm_dir):
    """TF (child process): build the classifier, load weights, export a SavedModel."""
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from resnet50_tf import build_resnet50
    model = build_resnet50(size=size, freeze_backbone=True)
    model.load_weights(weights)
    if Path(sm_dir).exists():
        shutil.rmtree(sm_dir)
    model.export(str(sm_dir))


def build_savedmodel_subprocess(weights, size, sm_dir):
    cmd = [sys.executable, __file__, "--_build_sm",
           "--weights", str(weights), "--size", str(size), "--sm-dir", str(sm_dir)]
    subprocess.run(cmd, check=True)


def convert_fp32(sm_dir, size, xml_out):
    """SavedModel -> genuinely FP32 IR.

    ov.save_model() defaults to compress_to_fp16=True, which silently halves the weight
    precision. On this classifier that drifts P(defective) by up to ~0.03 -- enough to flip
    patches that sit near the 0.5 decision threshold. Keep FP32 so the IR is numerically
    faithful to TF (max|diff| ~1e-6); the AI Suite compiler does its own calibration for the
    DLA anyway.
    """
    import openvino as ov
    ov_model = ov.convert_model(str(sm_dir), input=[[1, size, size, 3]])
    ov.save_model(ov_model, str(xml_out), compress_to_fp16=False)
    return xml_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help="trained .h5 weights")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--out", default=str(ROOT / "release_resnet"))
    ap.add_argument("--name", default="resnet50_pcb_fp32")
    # internal: SavedModel-builder child process
    ap.add_argument("--_build_sm", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--sm-dir", default="")
    args = ap.parse_args()

    if args._build_sm:
        build_savedmodel(args.weights, args.size, args.sm_dir)
        return

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sm = out / "_saved_model"
    print("Building SavedModel (subprocess) ...")
    build_savedmodel_subprocess(args.weights, args.size, str(sm))
    xml = out / f"{args.name}.xml"
    print(f"Converting to OpenVINO IR -> {xml}")
    convert_fp32(str(sm), args.size, xml)
    shutil.rmtree(sm, ignore_errors=True)
    (out / "classes.txt").write_text("good\nbad\n")
    print(f"Done. IR + classes.txt under {out}")
    print("Note: the AI Suite compiler can calibrate/quantize this FP32 IR for the DLA.")


if __name__ == "__main__":
    main()
