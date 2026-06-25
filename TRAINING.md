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

## Continue on your own NVIDIA GPU (e.g. H100) — Drive as sync hub

Google Drive is the shared state across machines (Colab / Mac / GPU box): the dataset zip
and the latest `*_yolov3_best.weights.h5` live there, so any machine pulls them, trains,
and pushes the new checkpoint back. The H100 has no Colab timeout, so a full run
(phase 1 + 2) finishes in well under an hour at `--batch 32`.

```bash
# 1. code
git clone https://github.com/EasonLi292/PCB_yolov3.git && cd PCB_yolov3

# 2. environment (GPU TF bundles CUDA; just needs a recent NVIDIA driver)
python3.12 -m venv .venv-train
./.venv-train/bin/pip install -r requirements-train.txt
./.venv-train/bin/python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 3. pull dataset + checkpoint from Drive  (rclone = headless; or scp from your Mac; or browser download)
#    one-time:  rclone config   (name the remote 'gdrive')
rclone copy gdrive:unified_pku_yolo_gray640.zip .
rclone copy gdrive:unified_pku_yolo_gray640_yolov3_best.weights.h5 runs/unified_pku_yolo_gray640/
mkdir -p datasets && unzip -q unified_pku_yolo_gray640.zip -d datasets

# 4. resume training (frozen continuation, then unfreeze) — H100 fits a big batch
./.venv-train/bin/python scripts/train_yolov3.py --data datasets/unified_pku_yolo_gray640 \
    --resume runs/unified_pku_yolo_gray640/unified_pku_yolo_gray640_yolov3_best.weights.h5 \
    --epochs 30 --batch 32
./.venv-train/bin/python scripts/train_yolov3.py --data datasets/unified_pku_yolo_gray640 \
    --resume runs/unified_pku_yolo_gray640/yolov3_best.weights.h5 --unfreeze --lr 1e-4 --epochs 20 --batch 32

# 5. export FPGA IR + push the trained checkpoint back to Drive
./.venv-train/bin/python scripts/export_fpga.py \
    --weights runs/unified_pku_yolo_gray640/yolov3_best.weights.h5 \
    --out runs/unified_pku_yolo_gray640/openvino_fpga --nc 6
rclone copy runs/unified_pku_yolo_gray640/yolov3_best.weights.h5 gdrive:
```

Alternatives to rclone for step 3: `scp` the files directly from your Mac
(`~/Downloads/PCB_yolov3/datasets/unified_pku_yolo_gray640` + the `.h5`), `gdown` for a
shared Drive link, or just regenerate the dataset on the box with the build/preprocess
scripts (needs the Kaggle/Roboflow creds again — slower).

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

## Stage 4 — FPGA AI Suite export (Intel Agilex 7 F-series) ([scripts/export_fpga.py](scripts/export_fpga.py))

For deployment on the FPGA DLA, export the **raw convolutional detection heads only** (3
outputs, no decode/NMS in the graph — pure conv/BN/LeakyReLU/upsample/concat). Decode +
NMS run on the host ([scripts/yolo_postprocess.py](scripts/yolo_postprocess.py), numpy).

```bash
./.venv-train/bin/python scripts/export_fpga.py \
    --weights runs/unified_pku_yolo_gray640/yolov3_best.weights.h5 \
    --out runs/unified_pku_yolo_gray640/openvino_fpga --nc 6 \
    --int8 --calib-data datasets/unified_pku_yolo_gray640/train --calib-n 300
```

Produces `yolov3_fpga_fp32.{xml,bin}` (and optional `--int8` → `yolov3_fpga_int8.{xml,bin}`).
Reads the `.h5` checkpoint directly, so an interrupted training run still exports. Validated
locally: the FP32 raw-output IR scores the same mAP@0.5 as the full decoded model (~0.41),
confirming host decode is equivalent.

**INT8:** for Agilex 7 / FPGA AI Suite, prefer feeding the **FP32 IR** to the AI Suite
compiler and letting `dla_compiler` calibrate INT8 (its DLA-tuned flow). The `--int8` NNCF
path is experimental — naive PTQ degraded accuracy badly on the early/undertrained weights
(it keeps the detection heads in FP32 but still needs a well-trained model + accuracy check).
Always re-run `analyze_openvino.py` on an INT8 IR before trusting it. Evaluate any IR with:

```bash
./.venv-train/bin/python scripts/analyze_openvino.py --ir .../yolov3_fpga_int8.xml \
    --data datasets/unified_pku_yolo_gray640 --split test --classes .../classes.txt
```
(`analyze_openvino.py` auto-detects 3-output FPGA IRs and decodes on the host.)

## Files
- [scripts/yolov3_tf.py](scripts/yolov3_tf.py) — model, loss, Darknet weight loader, tf.data pipeline.
- [scripts/train_yolov3.py](scripts/train_yolov3.py) — transfer-learning training CLI (`--resume`, `--unfreeze`).
- [scripts/export_openvino.py](scripts/export_openvino.py) — decoded IR + single-image inference (CPU demo).
- [scripts/export_fpga.py](scripts/export_fpga.py) — raw-output FPGA IR (FP32 + INT8/NNCF).
- [scripts/yolo_postprocess.py](scripts/yolo_postprocess.py) — host-side decode + NMS (numpy, ships with the FPGA runtime).
- [scripts/analyze_openvino.py](scripts/analyze_openvino.py) — test-set mAP / per-class AP / sample montage.

## Notes / recommended next steps
- **Anchors:** uses the standard COCO-derived YOLOv3 anchors. PCB defects are small;
  re-computing anchors (k-means over the dataset's box sizes) will improve recall.
- **Validation:** the `unified_pku_yolo` split is leakage-free (group-aware by board), so
  val numbers are meaningful. Consider using DeepPCB as an out-of-distribution test set.
- All stages were smoke-tested on Apple Silicon (build, transfer, forward/backward,
  SavedModel export, OpenVINO IR conversion, CPU inference).
