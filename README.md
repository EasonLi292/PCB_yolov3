# PCB Inspection — two models, one target

Two complementary PCB-inspection models, both exported to **OpenVINO IR for the Intel
FPGA AI Suite (Agilex 7 F-series SoC)**:

| Dir | Model | Task | Output |
|-----|-------|------|--------|
| [`yolov3/`](yolov3/) | YOLOv3 (Darknet-53) | **Detect & localize** every defect | boxes + 6 defect classes |
| [`resnet/`](resnet/) | ResNet-50 | **Classify** the whole board good vs. defective | `P(defective)` |

- **`yolov3/`** — the detector pipeline (build/preprocess datasets, transfer-learn,
  export FPGA IR, eval, report). See [`TRAINING.md`](TRAINING.md) and
  [`MODEL_REPORT.md`](MODEL_REPORT.md).
- **`resnet/`** — the board-level good/bad classifier. See [`resnet/README.md`](resnet/README.md),
  including where to download **defect-free "good board" images** (the scarce class).

Shared:
- [`datasets/`](datasets/) — normalized PCB-defect data (see [`datasets/README.md`](datasets/README.md)).
- `requirements-train.txt` — TensorFlow + OpenVINO env used by both.

Both target the same DLA. ResNet-50 is a reference model for the AI Suite (no custom
ops), so it deploys more simply than YOLO, which splits into raw conv heads + host-side
decode/NMS.
