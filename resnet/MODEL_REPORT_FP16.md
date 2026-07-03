# ResNet-50 PCB Classifier — Model Report (FP16 IR)

Same model and test set as [`MODEL_REPORT.md`](MODEL_REPORT.md), but measured on the
**FP16 OpenVINO IR** — the half-precision graph that maps to the Intel FPGA AI Suite
(Agilex 7 F-series) DLA. Output: `P(defective) ∈ [0, 1]` (**good = 0, defective = 1**).

> **Bottom line:** FP16 is effectively lossless — accuracy 0.9723 vs. FP32's 0.9725,
> only 5 of 5,264 predictions differ, and **zero defects newly missed**. Weight
> footprint halves (94 MB → 47 MB).

---

## 1. How this model differs from stock ResNet-50

Architecturally identical to the FP32 model — canonical ResNet-50 backbone (176 layers,
53 conv, 23,587,712 params matching `tf.keras.applications.ResNet50`), with the
1000-class ImageNet head swapped for `Dropout(0.3) → Dense(1, sigmoid)` = `P(defective)`.
See [`MODEL_REPORT.md §1`](MODEL_REPORT.md) for the full table and the two-phase
freeze/unfreeze training recipe.

**The only difference here is numeric precision:** the weights are stored as **float16**
(via OpenVINO `compress_to_fp16=True`) instead of float32. No layers, shapes, or
training change — this is purely the deployment IR.

* Input: `[1, 256, 256, 3]` · Output: `[1, 1]`
* `.bin` size: **47 MB** (FP32 is 94 MB)
* Exported from `best.weights.h5` → SavedModel → OpenVINO IR (`compress_to_fp16=True`).

---

## 2. Test-set results (FP16 OpenVINO runtime)

Held-out split = **templates 01 & 04**, **5,264 patches** (2,400 good / 2,864 defective).
Inference run through the OpenVINO CPU runtime with `INFERENCE_PRECISION_HINT=f16`.

### Confusion matrix @ threshold 0.50 (positive = defective)

|                | pred good | pred bad |
|----------------|-----------|----------|
| **actual good** | 2256 (TN) | 144 (FP) |
| **actual bad**  | 2 (FN)    | 2862 (TP) |

| metric | FP16 | (FP32 for reference) |
|---|---|---|
| accuracy | **0.9723** | 0.9725 |
| precision (defect) | 0.9521 | 0.9527 |
| recall (defect) — *bad boards caught* | **0.9993** | 0.9990 |
| F1 (defect) | 0.9751 | 0.9751 |
| ROC-AUC | 0.9995 | 0.999 |
| PR-AUC / average-precision | 0.9996 | 0.9996 |

Score separation is unchanged: mean `P(defective)` = **0.094** on good patches vs.
**0.997** on bad.

### Threshold sweep

| thr | acc | prec | recall |
|-----|-----|------|--------|
| 0.2 | 0.9411 | 0.9026 | 1.000 |
| 0.3 | 0.9576 | 0.9280 | 1.000 |
| 0.4 | 0.9654 | 0.9408 | 0.999 |
| **0.5** | **0.9723** | **0.9521** | **0.999** |
| 0.6 | 0.9785 | 0.9639 | 0.998 |
| 0.7 | 0.9821 | 0.9714 | 0.997 |
| 0.8 | 0.9854 | 0.9784 | 0.995 |

### FP32 → FP16 delta

- **5 / 5,264 predictions flip** (0.09%); mean |Δprob| ≈ 0.001, worst case 0.10.
- **0 defects newly missed.** Recall actually improves by one: the **spur** patch
  `tpl_01_001087` (FP32 `P=0.489`, just under the line) rounds to caught in FP16, so
  FN drops 3 → 2. The trade is 2 extra false alarms (FP 142 → 144) — a wash.
- Remaining FP16 false negatives (both still the tiny sub-frame defects):

  | patch idx | defect | size | P(defective) FP16 |
  |---|---|---|---|
  | 1531 | short | 48 × 54 px | 0.341 |
  | 2238 | open_circuit | 31 × 30 px | 0.060 |

### Takeaway

FP16 costs nothing measurable on this classifier — expected, since ResNet-50 is
well-conditioned and the good/bad scores are far from the 0.5 boundary. The precision
step that actually warrants a re-eval is **INT8**, which the AI Suite compiler can
calibrate for the DLA; that is where accuracy could move and should be re-measured.

*Reproduce:* export the FP16 IR (`resnet/export_openvino.py`), then run the OpenVINO
runtime over `datasets/pcb_patches/test` at `INFERENCE_PRECISION_HINT=f16`.
