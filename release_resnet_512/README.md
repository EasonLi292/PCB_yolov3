# PCB good/bad classifier — ResNet-50 @ 512×512 (deployment release)

Binary **good vs. defective** PCB patch classifier. Output: a single `P(defective) ∈ [0,1]`.
**Operating threshold = 0.50** → `P ≥ 0.50` means **defective**.

Target: Intel FPGA AI Suite DLA (Agilex 7). The whole graph is conv / BN / ReLU / add /
global-avg-pool / dense / sigmoid — no custom ops, so it maps to the DLA directly, with one
`[1,512,512,3]` input and one `[1,1]` output.

---

## Files

| file | size | what |
|---|---|---|
| `resnet50_pcb512_fp32.xml` | 147 KB | OpenVINO IR graph (**true FP32**) |
| `resnet50_pcb512_fp32.bin` | 94 MB | IR weights (FP32) |
| `resnet50_pcb512.weights.h5` | 95 MB | Keras inference weights (no optimizer state) |
| `classes.txt` | — | `good` / `bad` (index 0/1; the model emits P(bad)) |
| `test_dataset/` | 178 MB | held-out test set: `good/` 1,110 + `bad/` 1,184 = **2,294** patches |
| `ir_vs_tf_parity.json` | — | IR-vs-TensorFlow agreement on the full test set |

---

## Results at threshold 0.50 (held-out test set, 2,294 patches)

**Reference (TensorFlow, the repo's `eval_resnet.py`):**

|                | pred good | pred bad |
|----------------|-----------|----------|
| **actual good** | **1079** (TN) | **31** (FP) |
| **actual bad**  | **18** (FN) | **1166** (TP) |

| metric | value |
|---|---|
| **recall** (defects caught) | **0.985** |
| **accuracy** | **0.979** |
| precision | 0.974 |
| F1 | 0.979 |
| ROC-AUC | 0.996 |
| PR-AUC | 0.997 |

**As actually executed by this IR** (same 2,294 patches, OpenVINO CPU, deployment-style
OpenCV loader):

|                | pred good | pred bad |
|----------------|-----------|----------|
| **actual good** | 1075 (TN) | 35 (FP) |
| **actual bad**  | 20 (FN)   | 1164 (TP) |

recall **0.983** · accuracy **0.976** · precision 0.971

**The IR is faithful to TensorFlow:** on the full test set only **4 of 2,294 decisions differ**
(0.17%), mean |score difference| = 0.0016. The handful of differing patches sit right at the 0.50
boundary, where floating-point ordering and the JPEG-decode/resize path can tip a borderline score.
Expect ~±4 patches of wobble depending on your host preprocessing — not a model defect.

In plain terms: **it catches ~98.4% of defects and false-alarms on ~3% of clean patches.**

---

## Preprocessing — REQUIRED, not baked into the IR

The IR takes a raw `[1,512,512,3]` float tensor. You **must** apply the canonical Keras ResNet
preprocessing on the host first, or the output is meaningless:

1. Load image, resize to **512×512** (`cv2.INTER_AREA`).
2. Convert to **RGB**, cast to `float32` in the **0–255** range.
3. Apply `keras.applications.resnet50.preprocess_input` — i.e. **RGB→BGR** channel swap, then
   subtract the ImageNet BGR means `[103.939, 116.779, 123.68]`.

```python
import cv2, numpy as np, openvino as ov
from tensorflow.keras.applications.resnet50 import preprocess_input

core = ov.Core()
cm = core.compile_model("resnet50_pcb512_fp32.xml", "CPU")   # or the AI Suite DLA plugin
out = cm.output(0)

img = cv2.imread("patch.jpg")
img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
x = preprocess_input(img[None].copy())        # NOTE: preprocess_input mutates in place -> .copy()
p = float(cm(x)[out].ravel()[0])              # P(defective)
verdict = "bad" if p >= 0.50 else "good"
```

> ⚠️ `preprocess_input` **modifies its input array in place**. Always pass a `.copy()`, and never
> apply it twice — double-preprocessing silently produces confident, wrong answers.

---

## Model / training provenance

- Architecture: stock **ResNet-50** backbone (ImageNet init, fully fine-tuned) with the 1000-class
  head replaced by `Dropout(0.3) → Dense(1, sigmoid)`. 23.6 M params.
- Two-phase transfer: frozen head → full fine-tune. **~30 min** GPU wall-time.
- Input **512×512**; trained with defects placed **randomly anywhere** in the tile
  (`--defect-offset 0.4`), so it does not assume the defect is centered.
- Source weights: `runs_resnet_v3/pcb_bin_offset_512/best.weights.h5` (md5 `fb6bf9ad…`).
- Data: HRIPCB-mined patches. **good** = crops of the per-pixel-median "healed" clean plate;
  **bad** = crops around annotated defects. Split is a leak-free per-(photo, defect) unit holdout —
  all 10 board designs appear in train and test (the fixed-production-line, in-distribution regime).

### Known caveats
1. **Patch-level, not board-level.** A full board is scanned as many sliding-window patches, so
   board-level recall will be *higher* (several chances to catch a defect) and board-level false
   alarms *higher* too (more patches, more chances to fire). A board-level aggregator is not part
   of this release.
2. **`good` comes from healed plates**, so "detects defects" and "detects healing artifacts" are
   not fully separable; a true check needs real defect-free photographs.
3. **Sensor noise matters.** Above ~σ10 gray-levels of read noise, accuracy degrades sharply (at
   512 more than at 256, since 512 feeds fine detail — and noise — straight in). Keep the rig
   under ~σ5.

*Threshold is a dial, not a constant* — the sweep on the same test set:

| threshold | accuracy | precision | recall |
|---|---|---|---|
| 0.30 | 0.940 | 0.902 | 0.991 |
| 0.40 | 0.968 | 0.951 | 0.988 |
| **0.50** | **0.979** | **0.974** | **0.985** |
| 0.60 | 0.981 | 0.985 | 0.979 |
| 0.70 | 0.983 | 0.992 | 0.974 |

Lower it to catch more defects at the cost of more false alarms.
