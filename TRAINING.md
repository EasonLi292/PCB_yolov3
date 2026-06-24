# Training YOLOv3 on the PCB defect datasets

Three-stage pipeline: **train (TensorFlow) → export → run (OpenVINO)**. OpenVINO is the
inference/optimization runtime, *not* a training framework, so it only appears in stage 3.

## Why a second venv

The main `.venv` uses Python 3.14, which **TensorFlow does not yet support**. Training
uses a separate Python 3.12 venv:

```bash
python3.12 -m venv .venv-train
./.venv-train/bin/pip install "tensorflow==2.21.0" "openvino>=2025.0" opencv-python-headless numpy
```

## Stage 1–2 — train with transfer learning ([scripts/train_yolov3.py](scripts/train_yolov3.py))

YOLOv3 = Darknet-53 backbone + 3-scale FPN heads. We:
1. Load the official pretrained `yolov3.weights` (COCO, 80 classes).
2. **Transfer** the backbone + the class-independent FPN necks.
3. **Redefine the detection heads** (`yolo_output_*`) for our class count — 33 filters
   per head (`3 anchors × (5 + 6 classes)`) — trained from scratch.
4. Freeze the backbone, train necks + new heads (then optionally `--unfreeze` to
   fine-tune the whole network at a low LR).

```bash
# downloads pretrained weights on first run (237 MB)
./.venv-train/bin/python scripts/train_yolov3.py \
    --data datasets/unified_pku_yolo_gray640 \
    --download-weights --epochs 50 --batch 8

# sanity check without training (1 step on 1 batch)
./.venv-train/bin/python scripts/train_yolov3.py --data datasets/unified_pku_yolo_gray640 --smoke
```

Outputs to `runs/<dataset>/`: best checkpoint (`.weights.h5`), an NMS-free inference
`saved_model/`, and `classes.txt`. Works on any of the built datasets — pass
`datasets/dspcbsd_yolo_gray640` (9 classes) or the color `*_yolo/` sets too.

**Grayscale input:** the `*_gray640` images are single-channel; the data pipeline tiles
them to 3 channels so the pretrained RGB backbone stays valid.

## Stage 3 — export to OpenVINO + run inference ([scripts/export_openvino.py](scripts/export_openvino.py))

```bash
./.venv-train/bin/python scripts/export_openvino.py \
    --saved-model runs/unified_pku_yolo_gray640/saved_model \
    --out runs/unified_pku_yolo_gray640/openvino \
    --image datasets/unified_pku_yolo_gray640/test/images/<some>.png \
    --classes runs/unified_pku_yolo_gray640/classes.txt --score 0.25
```

Produces OpenVINO IR (`yolov3.xml` / `.bin`) and a drawn prediction image. The network is
exported **without** NMS (TF's `CombinedNonMaxSuppression` has no OpenVINO conversion
rule); NMS runs as a `cv2.dnn.NMSBoxes` post-process. For INT8 speedup, quantize the IR
later with NNCF.

## Files
- [scripts/yolov3_tf.py](scripts/yolov3_tf.py) — model, loss, Darknet weight loader, tf.data pipeline.
- [scripts/train_yolov3.py](scripts/train_yolov3.py) — transfer-learning training CLI.
- [scripts/export_openvino.py](scripts/export_openvino.py) — IR conversion + inference.

## Notes / recommended next steps
- **Anchors:** uses the standard COCO-derived YOLOv3 anchors. PCB defects are small;
  re-computing anchors (k-means over the dataset's box sizes) will improve recall.
- **Validation:** the `unified_pku_yolo` split is leakage-free (group-aware by board), so
  val numbers are meaningful. Consider using DeepPCB as an out-of-distribution test set.
- All stages were smoke-tested on Apple Silicon (build, transfer, forward/backward,
  SavedModel export, OpenVINO IR conversion, CPU inference).
