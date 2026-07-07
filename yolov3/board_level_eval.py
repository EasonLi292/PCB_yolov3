#!/usr/bin/env python3
"""
Goal 1 support: BOARD-LEVEL good/bad evaluation of the YOLOv3 detector, to compare with
the ResNet good/bad classifier on the same terms.

Standard mAP measures detection quality on defective boards; it never tells you how often
the detector falsely fires on a perfectly good board. This script does:

  * DEFECTIVE boards (the test split): board = BAD if the detector emits >=1 detection
    above --score. Fraction flagged = board-level RECALL (defects caught).
  * GOOD boards (guaranteed defect-free: DeepPCB *_temp templates + HRIPCB healed clean
    plates): board = BAD if any detection. Fraction flagged = FALSE-ALARM rate.

Reports the board-level confusion matrix, accuracy, precision, recall, and false-alarm
rate -- the numbers to line up against resnet/MODEL_REPORT.md.

Inference path is identical to analyze_openvino.py (FPGA raw-heads -> host decode, or the
decode-baked IR), so results match the deployment path.

Example:
  python yolov3/board_level_eval.py \
      --ir runs/unified_pku_yolo_gray640/openvino_fpga/yolov3_fpga_fp32.xml \
      --defective datasets/unified_pku_yolo_gray640 --split test \
      --good "datasets/deeppcb/PCBData/**/*_temp.jpg" "datasets/clean_plates/plate_*.png" \
      --classes release/classes.txt --score 0.25
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np
import openvino as ov

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_openvino import preprocess, postprocess   # same inference path as analyze_openvino


def n_detections(compiled, fpga_raw, decode_and_nms, img_path, size, score, iou):
    """Run the detector on one image; return the number of detections above threshold."""
    inp, _vis, (h0, w0) = preprocess(img_path, size)
    result = compiled(inp)
    if fpga_raw:
        raw = [result[o] for o in compiled.outputs]
        dets = decode_and_nms(raw, w0, h0, score, iou)
    else:
        pred = result[compiled.output(0)][0]
        dets = postprocess(pred, w0, h0, score, iou)
    return len(dets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", required=True, help="trained OpenVINO IR (.xml)")
    ap.add_argument("--defective", required=True, help="dataset dir with <split>/images (all defective)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--good", nargs="+", required=True,
                    help="one or more globs of guaranteed-good board images")
    ap.add_argument("--classes", required=True)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--score", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="cap boards per group (0 = all)")
    args = ap.parse_args()

    compiled = ov.Core().compile_model(args.ir, "CPU")
    fpga_raw = len(compiled.outputs) >= 3
    decode_and_nms = None
    if fpga_raw:
        from yolo_postprocess import decode_and_nms
    print(f"IR outputs: {len(compiled.outputs)} ->",
          "FPGA raw heads (host decode)" if fpga_raw else "decoded IR")

    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    defective = sorted(p for p in (Path(args.defective) / args.split / "images").iterdir()
                       if p.suffix.lower() in exts)
    good = []
    for g in args.good:
        good += [Path(p) for p in glob.glob(g, recursive=True)]
    good = sorted(set(good))
    if args.limit:
        defective, good = defective[:args.limit], good[:args.limit]
    print(f"defective boards: {len(defective)}   good boards: {len(good)}")

    def flagged(paths):
        f = 0
        for i, p in enumerate(paths):
            if n_detections(compiled, fpga_raw, decode_and_nms, p, args.size, args.score, args.iou) > 0:
                f += 1
            if (i + 1) % 200 == 0:
                print(f"  ...{i+1}/{len(paths)}")
        return f

    print("scoring defective boards..."); bad_flagged = flagged(defective)
    print("scoring good boards...");      good_flagged = flagged(good)

    nD, nG = len(defective), len(good)
    TP = bad_flagged                 # defective correctly flagged bad
    FN = nD - bad_flagged            # defective missed (called good) -- DANGEROUS
    FP = good_flagged                # good boards falsely flagged -- FALSE ALARMS
    TN = nG - good_flagged           # good correctly passed
    acc = (TP + TN) / max(nD + nG, 1)
    recall = TP / nD if nD else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    fa_rate = FP / nG if nG else 0.0

    print(f"\n== BOARD-LEVEL good/bad @ score {args.score} ==")
    print(f"                pred good   pred bad")
    print(f"  defective     {FN:8d}   {TP:8d}   (recall {recall:.3f})")
    print(f"  good          {TN:8d}   {FP:8d}   (false-alarm {fa_rate:.3f})")
    print(f"\naccuracy            : {acc:.3f}")
    print(f"precision (bad)     : {precision:.3f}")
    print(f"recall (defects)    : {recall:.3f}   <- fraction of defective boards caught")
    print(f"false-alarm rate    : {fa_rate:.3f}  <- fraction of GOOD boards wrongly flagged")
    print(f"good boards passed  : {TN}/{nG}")
    print("\nCompare these board-level numbers with resnet/MODEL_REPORT.md "
          "(sweep --score to trade recall vs false alarms).")


if __name__ == "__main__":
    main()
