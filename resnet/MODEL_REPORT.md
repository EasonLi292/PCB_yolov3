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

Held-out split = **templates 01 & 04** (board layouts never seen in training),
**5,264 patches** (2,400 good / 2,864 defective).

### Confusion matrix @ threshold 0.50 (positive = defective)

|                | pred good | pred bad |
|----------------|-----------|----------|
| **actual good** | 2258 (TN) | 142 (FP) |
| **actual bad**  | 3 (FN)    | 2861 (TP) |

| metric | value |
|---|---|
| accuracy | **0.972** |
| precision (defect) | 0.953 |
| recall (defect) — *bad boards caught* | **0.999** |
| F1 (defect) | 0.975 |
| ROC-AUC | 0.999 |
| PR-AUC / average-precision | 0.9996 |

Score separation is near-clean: mean `P(defective)` = **0.09** on good patches vs.
**0.997** on bad.

### Threshold sweep

| thr | acc | prec | recall |
|-----|-----|------|--------|
| 0.2 | 0.942 | 0.904 | 1.000 |
| 0.3 | 0.958 | 0.929 | 1.000 |
| 0.4 | 0.966 | 0.941 | 0.999 |
| **0.5** | **0.972** | **0.953** | **0.999** |
| 0.6 | 0.979 | 0.964 | 0.998 |
| 0.7 | 0.982 | 0.971 | 0.997 |
| 0.8 | 0.985 | 0.978 | 0.995 |

### Where the errors are

- **142 false positives** (good → bad): clean regions flagged anyway; 128 on template
  01, 14 on template 04. Failure direction is *over-cautious*, which is the safe side
  for a screen. Sample montage: [`test_errors.jpg`](test_errors.jpg).
- **3 false negatives** (defect → good) — all on template 01, all among the smallest
  defects, boxed in [`false_negatives.jpg`](false_negatives.jpg):

  | patch idx | defect | size | P(defective) |
  |---|---|---|---|
  | 1087 | spur | 38 × 38 px | 0.489 (just under 0.5) |
  | 1531 | short | 48 × 54 px | 0.337 |
  | 2238 | open_circuit | 31 × 30 px | 0.061 (confident miss) |

  Same failure mode: a ~30 px defect cropped into the 1024 px window and stored at 384
  becomes ~1 % of the frame, surrounded by high-contrast pins/vias. Dropping the
  threshold to 0.4 recovers idx 1087; idx 2238 needs more defect signal (tighter
  `--patch` / larger `--save-size`), not a threshold change.

*Reproduce:* `python resnet/eval_resnet.py --weights best.weights.h5 --data datasets/pcb_patches --size 256`
