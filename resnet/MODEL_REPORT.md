# ResNet-50 PCB Classifier — Model Report

Board-patch **good vs. defective** classifier. Output: `P(defective) ∈ [0, 1]`
(label convention **good = 0, defective = 1**). See [`README.md`](README.md) for the
data pipeline; this doc covers **(1) how it differs from stock ResNet-50** and
**(2) the test-set confusion matrix**.

---

## 1. How this model differs from stock ResNet-50

Architecturally it **is** canonical ResNet-50 — the convolutional feature extractor is
untouched. Verified against `tf.keras.applications.ResNet50(weights="imagenet")`:

| | Stock ResNet-50 (ImageNet) | This model |
|---|---|---|
| Backbone (conv/BN/ReLU/add/global-avg-pool) | 176 layers, 53 conv, **23,587,712** params | **identical**, same layers & param count |
| Backbone weights | ImageNet | ImageNet (kept, then fine-tuned) |
| Input | 224 × 224 × 3 | **256 × 256 × 3** (global-avg-pool is size-agnostic) |
| Final layer | Dense **1000** + softmax (ImageNet classes) | Dropout(0.3) → Dense **1** + **sigmoid** |
| Output | 1000-way class probabilities | single `P(defective)` |
| Loss | categorical cross-entropy | binary cross-entropy + inverse-freq class weights |

**In one sentence:** it's ResNet-50 with the 1000-class head swapped for one sigmoid
neuron — everything else (all residual blocks, batch-norm, global average pooling) is
the stock ImageNet network. Full model = 23,589,761 params (backbone + 2,049-param head).

### Training recipe (two-phase transfer, like the YOLO detector)
1. **Phase 1 — freeze:** backbone frozen, only the head trains (2,049 trainable params).
2. **Phase 2 — unfreeze:** whole network fine-tuned at low LR (23,536,641 trainable) to
   adapt the ImageNet features to the PCB domain.

### Why ResNet-50 (deployment)
It's a first-class **reference model for the Intel FPGA AI Suite DLA** (plain
conv/BN/ReLU/add/pool/dense, no custom ops), so the graph maps to the Agilex 7 target
with a single `[1,256,256,3] → [1,1]` export — simpler to deploy than YOLO's raw conv
heads + host-side decode/NMS.

---

## 2. Test-set results

Measured on the actual trained dataset `datasets/pcb_patches` (**FP32**, Keras). The
split is ~8/1/1 at the **patch level** — all 10 templates appear in every split, so this
is an *in-distribution* test (performance on **boards the model has seen**), which
matches the deployment plan of training on the same boards you inspect. Test split =
**2,384 patches** (1,200 good / 1,184 defective).

### Confusion matrix @ threshold 0.50 (positive = defective)

|                | pred good | pred bad |
|----------------|-----------|----------|
| **actual good** | 1159 (TN) | 41 (FP) |
| **actual bad**  | 4 (FN)    | 1180 (TP) |

| metric | value |
|---|---|
| accuracy | **0.9811** |
| precision (defect) | 0.9664 |
| recall (defect) — *bad boards caught* | **0.9966** |
| F1 (defect) | 0.9813 |
| ROC-AUC | 0.9995 |
| PR-AUC / average-precision | 0.9995 |

Score separation is near-clean: mean `P(defective)` = **0.059** on good patches vs.
**0.994** on bad.

### Threshold sweep

| thr | acc | prec | recall |
|-----|-----|------|--------|
| 0.2 | 0.9568 | 0.9206 | 0.9992 |
| 0.3 | 0.9702 | 0.9441 | 0.9992 |
| 0.4 | 0.9782 | 0.9594 | 0.9983 |
| **0.5** | **0.9811** | **0.9664** | **0.9966** |
| 0.6 | 0.9845 | 0.9728 | 0.9966 |
| 0.7 | 0.9866 | 0.9800 | 0.9932 |
| 0.8 | 0.9874 | 0.9849 | 0.9899 |

### Where the errors are

- **41 false positives** (good → bad): clean regions flagged anyway. Failure direction
  is *over-cautious* — 41 false alarms vs. only 4 missed defects, the safe side for a
  screen. Sample montage: [`test_errors.jpg`](test_errors.jpg).
- **4 false negatives** (defect → good), all near/under the 0.5 line, shown with their
  scores in [`false_negatives.jpg`](false_negatives.jpg) (bad patches are cropped
  centered on the defect):

  | patch | defect region | P(defective) |
  |---|---|---|
  | `tpl_01_bad_u00128_v3` | spur near dense routing | 0.489 (just under 0.5) |
  | `tpl_08_bad_u01867_v0` | LED-array trace | 0.351 |
  | `tpl_08_bad_u01638_v3` | LED-array trace | 0.432 |
  | `tpl_08_bad_u01638_v0` | LED-array trace | 0.066 (confident miss) |

  Same failure mode as before: the smallest defects, downscaled into a busy frame, sit
  near the visibility floor. Dropping the threshold to 0.4 recovers the two 0.43–0.49
  cases; the 0.066 miss needs more defect signal (tighter crop / larger `--save-size`),
  not a threshold change.

### A note on leakage (minor, accepted)

Each defect is minted as 4 jittered siblings (`u<id>_v0..v3`, ±12 px / ±12°); the
per-patch split scatters them across train/test, so ~14 % of test-**bad** patches have a
near-identical twin the model trained on (the **good** patches are clean — ~0 %
near-dups). This slightly inflates bad-class recall/PR-AUC; good-side precision and false
alarms are unaffected. Honest range is ~0.97–0.98 (an earlier by-template held-out split
scored 0.972). Acceptable here since the deployment target is the same-boards regime; for
a leakage-free figure, split per-defect instead of per-patch.

*Reproduce:* `python resnet/eval_resnet.py --weights best.weights.h5 --data datasets/pcb_patches --size 256`
