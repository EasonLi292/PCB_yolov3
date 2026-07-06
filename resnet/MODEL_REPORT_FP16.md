# ResNet-50 PCB Classifier — Model Report (FP16 IR)

Same model and test set as [`MODEL_REPORT.md`](MODEL_REPORT.md), but measured on the
**FP16 OpenVINO IR** — the half-precision graph that maps to the Intel FPGA AI Suite
(Agilex 7 F-series) DLA. Output: `P(defective) ∈ [0, 1]` (**good = 0, defective = 1**).

> **Bottom line:** FP16 is effectively lossless — accuracy 0.9811, identical to FP32,
> only 2 of 2,384 predictions differ, and **zero defects newly missed**. Weight
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

Same in-distribution split as [`MODEL_REPORT.md §2`](MODEL_REPORT.md) (all 10 templates
in every split), test = **2,384 patches** (1,200 good / 1,184 defective). Inference run
through the OpenVINO CPU runtime with `INFERENCE_PRECISION_HINT=f16`.

### Confusion matrix @ threshold 0.50 (positive = defective)

|                | pred good | pred bad |
|----------------|-----------|----------|
| **actual good** | 1158 (TN) | 42 (FP) |
| **actual bad**  | 3 (FN)    | 1181 (TP) |

| metric | FP16 | (FP32 for reference) |
|---|---|---|
| accuracy | **0.9811** | 0.9811 |
| precision (defect) | 0.9657 | 0.9664 |
| recall (defect) — *bad boards caught* | **0.9975** | 0.9966 |
| F1 (defect) | 0.9813 | 0.9813 |
| ROC-AUC | 0.9995 | 0.9995 |
| PR-AUC / average-precision | 0.9995 | 0.9995 |

Score separation is unchanged: mean `P(defective)` = **0.060** on good patches vs.
**0.994** on bad.

### Threshold sweep

| thr | acc | prec | recall |
|-----|-----|------|--------|
| 0.2 | 0.9568 | 0.9206 | 0.9992 |
| 0.3 | 0.9702 | 0.9441 | 0.9992 |
| 0.4 | 0.9782 | 0.9594 | 0.9983 |
| **0.5** | **0.9811** | **0.9657** | **0.9975** |
| 0.6 | 0.9849 | 0.9736 | 0.9966 |
| 0.7 | 0.9857 | 0.9784 | 0.9932 |
| 0.8 | 0.9874 | 0.9857 | 0.9890 |

### FP32 → FP16 delta

- **2 / 2,384 predictions flip** (0.08%); mean |Δprob| ≈ 0.001.
- **0 defects newly missed.** Recall actually improves by one: the borderline spur patch
  `tpl_01_bad_u00128_v3` (FP32 `P=0.489`, just under the line) rounds to caught in FP16,
  so FN drops 4 → 3. The trade is 1 extra false alarm (FP 41 → 42) — a wash.
- Remaining FP16 false negatives (still the smallest sub-frame defects):

  | patch | P(defective) FP16 |
  |---|---|
  | `tpl_08_bad_u01638_v3` | 0.421 |
  | `tpl_08_bad_u01867_v0` | 0.355 |
  | `tpl_08_bad_u01638_v0` | 0.065 (confident miss) |

### Takeaway

FP16 costs nothing measurable on this classifier — expected, since ResNet-50 is
well-conditioned and the good/bad scores are far from the 0.5 boundary. The precision
step that actually warrants a re-eval is **INT8**, which the AI Suite compiler can
calibrate for the DLA; that is where accuracy could move and should be re-measured.

*Reproduce:* export the FP16 IR (`resnet/export_openvino.py`), then run the OpenVINO
runtime over `datasets/pcb_patches/test` at `INFERENCE_PRECISION_HINT=f16`.
