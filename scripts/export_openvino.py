#!/usr/bin/env python3
"""
Export a trained YOLOv3 TF SavedModel to OpenVINO IR and run inference.

Stage 3 of the pipeline (train_yolov3.py is stages 1-2). OpenVINO is the
inference/optimization runtime -- it converts the trained graph to its IR format
(.xml/.bin) and runs optimized CPU inference. Optional INT8 quantization can be added
later with NNCF.

Examples:
  # convert only
  python scripts/export_openvino.py --saved-model runs/unified_pku_yolo_gray640/saved_model \
         --out runs/unified_pku_yolo_gray640/openvino
  # convert + run inference on an image and save a visualization
  python scripts/export_openvino.py --saved-model .../saved_model --out .../openvino \
         --image datasets/unified_pku_yolo_gray640/test/images/hr_01_short_05.png \
         --classes runs/unified_pku_yolo_gray640/classes.txt
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
import openvino as ov


def convert(saved_model_dir: Path, out_dir: Path, size: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting {saved_model_dir} -> OpenVINO IR ...")
    ov_model = ov.convert_model(str(saved_model_dir),
                                input=[[1, size, size, 3]])
    xml = out_dir / "yolov3.xml"
    ov.save_model(ov_model, str(xml))
    print(f"Saved IR: {xml} (+ .bin)")
    return xml


def preprocess(image_path: Path, size: int):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)   # black & white
    h0, w0 = img.shape[:2]
    img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)              # tile to 3ch
    resized = cv2.resize(img3, (size, size)).astype(np.float32) / 255.0
    return resized[None, ...], img3, (h0, w0)


def postprocess(pred, w0, h0, score_thresh, iou_thresh):
    """pred: (N, 4+1+classes) = [x1,y1,x2,y2 (norm), obj, *class_probs]. NMS via cv2."""
    boxes_n = pred[:, :4]
    obj = pred[:, 4]
    class_probs = pred[:, 5:]
    cls_id = class_probs.argmax(axis=1)
    cls_score = class_probs.max(axis=1)
    scores = obj * cls_score
    keep = scores >= score_thresh
    boxes_n, scores, cls_id = boxes_n[keep], scores[keep], cls_id[keep]
    if len(scores) == 0:
        return []
    # to pixel xywh for cv2.dnn.NMSBoxes
    rects = [[int(x1 * w0), int(y1 * h0), int((x2 - x1) * w0), int((y2 - y1) * h0)]
             for x1, y1, x2, y2 in boxes_n]
    idxs = cv2.dnn.NMSBoxes(rects, scores.tolist(), score_thresh, iou_thresh)
    dets = []
    for i in np.array(idxs).flatten():
        x, y, w, h = rects[i]
        dets.append((x, y, x + w, y + h, float(scores[i]), int(cls_id[i])))
    return dets


def infer(xml: Path, image_path: Path, size: int, classes, score_thresh, out_dir: Path,
          iou_thresh=0.45):
    core = ov.Core()
    compiled = core.compile_model(str(xml), "CPU")
    inp, vis, (h0, w0) = preprocess(image_path, size)
    pred = compiled(inp)[compiled.output(0)][0]    # (N, 4+1+classes)
    dets = postprocess(pred, w0, h0, score_thresh, iou_thresh)
    print(f"{image_path.name}: {len(dets)} detections (score >= {score_thresh})")
    for x1, y1, x2, y2, s, c in dets:
        label = classes[c] if classes else str(c)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, f"{label} {s:.2f}", (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        print(f"  {label:16s} {s:.3f}  [{x1},{y1},{x2},{y2}]")
    out_img = out_dir / f"pred_{image_path.stem}.jpg"
    cv2.imwrite(str(out_img), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization: {out_img}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--image", help="optional image to run inference on")
    ap.add_argument("--classes", help="classes.txt (one name per line)")
    ap.add_argument("--score", type=float, default=0.25)
    args = ap.parse_args()

    out_dir = Path(args.out)
    xml = convert(Path(args.saved_model), out_dir, args.size)

    if args.image:
        classes = None
        if args.classes and Path(args.classes).exists():
            classes = Path(args.classes).read_text().split()
        infer(xml, Path(args.image), args.size, classes, args.score, out_dir)


if __name__ == "__main__":
    main()
