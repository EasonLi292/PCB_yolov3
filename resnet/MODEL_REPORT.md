# ResNet-50 PCB Classifier — Model Report

Board-patch **good vs. defective** classifier. Output: `P(defective) ∈ [0, 1]`
(label convention **good = 0, defective = 1**). See [`README.md`](README.md) for the data
pipeline. This doc covers **(1) how it differs from stock ResNet-50** and **(2) the 2×2
experiment grid** — input resolution (256 vs 512) × defect placement (centered vs random-offset) —
with confusion matrices, confidence separation, threshold sweeps, and imaging-noise robustness.

All numbers are the **v3 in-distribution** run: freshly-mined datasets, a leak-free
per-(photo, defect) unit split, defects placed randomly for the "offset" columns. Test split =
**2,294 patches** (1,110 good / 1,184 defective) for every config.

---

## 1. How this model differs from stock ResNet-50

Architecturally it **is** canonical ResNet-50 — the convolutional feature extractor is untouched.
Verified against `tf.keras.applications.ResNet50(weights="imagenet")`:

| | Stock ResNet-50 (ImageNet) | This model |
|---|---|---|
| Backbone (conv/BN/ReLU/add/global-avg-pool) | 176 layers, 53 conv, **23,587,712** params | **identical** |
| Backbone weights | ImageNet | ImageNet (kept, then fine-tuned) |
| Input | 224 × 224 × 3 | **256 or 512 × 3** (global-avg-pool is size-agnostic) |
| Final layer | Dense **1000** + softmax | Dropout(0.3) → Dense **1** + **sigmoid** |
| Loss | categorical cross-entropy | binary cross-entropy + inverse-freq class weights |

**In one sentence:** ResNet-50 with the 1000-class head swapped for one sigmoid neuron —
everything else is the stock ImageNet network. Two-phase transfer (freeze head → unfreeze
backbone at low LR). First-class **reference model for the Intel FPGA AI Suite DLA** (plain
conv/BN/ReLU/add/pool/dense), exported `[1, size, size, 3] → [1, 1]`.

---

## 2. The experiment grid

Four models, one recipe, two levers. **Centered** = defect at tile center (easy, unrealistic);
**offset** = defect placed randomly up to 0.4·patch off-center (simulates a real board where the
defect can be anywhere). Best-threshold accuracy:

| | **256 px** | **512 px** |
|---|---|---|
| **centered** | 0.963 | **0.996** |
| **offset** (random placement) | 0.936 | **0.984** |

Two readings:
- **Resolution buys the most.** 512 beats 256 by +0.033 (centered) / +0.048 (offset).
- **Random placement costs 256 more than 512.** Offset drops 256 by 0.027 but 512 by only 0.012 —
  higher resolution absorbs position variation almost completely.

This replicates last week's ~0.97 (256-centered = 0.963) **without** the leak that inflated it,
and the offset columns are the honest "defect could be anywhere" numbers.

### Training cost (GPU wall-time, two-phase 30+25 epochs)

| model | input | batch | wall-time |
|---|---|---|---|
| baseline (256 centered) | 256 | 64 | **~16 min** |
| 256 offset | 256 | 64 | ~17 min |
| 512 centered | 512 | 16 | ~36 min |
| 512 offset (deployment) | 512 | 16 | **~30 min** |

512 costs ~2× the wall-time of 256 (4× the pixels, but the smaller batch offsets some of it).
Even the slowest ResNet trains in half an hour — cheap next to the YOLO detector's 2 h 28 m.

---

## 3. Confusion matrices @ threshold 0.50 (positive = defective)

**256 centered** — acc 0.957 · prec 0.950 · recall 0.969 · ROC-AUC 0.995

|                | pred good | pred bad |
|----------------|-----------|----------|
| **actual good** | 1049 | 61 |
| **actual bad**  | 37 | 1147 |

**256 offset** — acc 0.912 · prec 0.899 · recall 0.933 · ROC-AUC 0.971

|                | pred good | pred bad |
|----------------|-----------|----------|
| **actual good** | 986 | 124 |
| **actual bad**  | 79 | 1105 |

**512 centered** — acc 0.994 · prec 0.994 · recall 0.995 · ROC-AUC 1.000

|                | pred good | pred bad |
|----------------|-----------|----------|
| **actual good** | 1103 | 7 |
| **actual bad**  | 6 | 1178 |

**512 offset** — acc 0.979 · prec 0.974 · recall 0.985 · ROC-AUC 0.996

|                | pred good | pred bad |
|----------------|-----------|----------|
| **actual good** | 1079 | 31 |
| **actual bad**  | 18 | 1166 |

Failure direction is **over-cautious** in every config (false alarms ≥ misses) — the safe side for
a screen. The 512 models miss only 6–18 of 1,184 defects.

---

## 4. Operating threshold — recall-leaning, but accuracy still high

A missed defect (FN → a bad board shipped) is the expensive error, so we lean the threshold toward
recall — but only as far as keeps overall accuracy high. The principled point for "recall matters
more than precision, but both count" is the **F2-optimal** threshold (F2 weights recall 2× vs.
precision). Chasing an arbitrary recall target (e.g. 0.99) instead is a mistake: on these models it
buys a sliver of recall for a large accuracy hit.

| model | @0.50 (recall / acc) | **F2-optimal** (thr, recall, acc, FA) |
|---|---|---|
| 256 centered | 0.969 / 0.957 | **0.34 · 0.982 · 0.953 · 7.8%** |
| 256 offset | 0.933 / 0.912 | 0.49 · 0.938 · 0.907 · 12.6% |
| 512 centered | 0.995 / 0.994 | 0.60 · 0.995 · 0.996 · 0.4% |
| **512 offset** (deployment) | 0.985 / 0.979 | **0.48 · 0.986 · 0.978 · 3.1%** |

**The key result: for the 512 models the balanced optimum is essentially 0.5 already.** They are so
well-separated that the default threshold gives *both* great recall (0.985–0.995) and high accuracy
(0.978–0.996) — the F2 optimum barely moves it (to 0.48–0.60). So the recommendation is simply:

- **512 offset (deployment): keep the threshold at ≈ 0.48–0.50** → recall **0.986**, accuracy
  **0.978** (17–18 of 1,184 defects missed, ~3% false alarms). That *is* the "recall great,
  accuracy high" point; do **not** push lower — dropping to 0.37 would gain only ~0.005 recall while
  costing ~2 points of accuracy, and forcing zero misses costs **79%** false alarms.
- **256 centered: nudge to 0.34** — the one model that benefits, lifting recall 0.969 → **0.982**
  for a negligible 0.4-point accuracy change.
- **256 offset stays at ~0.5** and remains the weakest model (accuracy only ~0.91); if misses
  matter, use 512, whose resolution is what buys recall *and* accuracy at once.

*(Patch-level; a defective board is caught if any of its patches fire, so board-level recall runs
higher than these per-patch numbers.)* Source: `resnet/details/balanced_threshold.json` (F2 sweep)
and `threshold_pick.json` (recall-target detail).

---

## 5. Confidence separation

Mean `P(defective)` on good vs. defective patches, and how many land in the contested 0.1–0.9 band
(the only region where errors occur):

| config | good mean | bad mean | good < 0.1 | bad > 0.9 | contested (good / bad) |
|---|---|---|---|---|---|
| 256 centered | 0.076 | 0.965 | 82.8% | 93.5% | 251 (180 / 71) |
| 256 offset | 0.215 | 0.926 | 41.5% | 86.3% | **791** (645 / 146) |
| **512 centered** | **0.023** | **0.994** | **95.1%** | **99.0%** | **63** (53 / 10) |
| 512 offset | 0.127 | 0.981 | 60.6% | 96.4% | 477 (435 / 42) |

The models are **decisive, not hedging** — most patches pile against 0 or 1. Random placement
widens the *good* distribution (its mean climbs from 0.076→0.215 at 256, 0.023→0.127 at 512): a
defect at the tile edge makes some BAD tiles mostly-clean, so the model grows warier of genuinely
clean tiles. In every config the uncertainty is **overwhelmingly on the good side** (contested
band is ~3–4× more good than bad), i.e. the model second-guesses clean patches, not defective ones.
512-centered is near-perfect: only 63 of 2,294 patches are contested.

### Threshold sweep (accuracy / precision / recall)

| thr | 256 ctr | 256 off | 512 ctr | 512 off |
|---|---|---|---|---|
| 0.3 | 0.949 / 0.923 / 0.982 | 0.825 / 0.763 / 0.959 | 0.993 / 0.990 / 0.996 | 0.940 / 0.902 / 0.991 |
| **0.5** | 0.957 / 0.950 / 0.969 | 0.912 / 0.899 / 0.933 | 0.994 / 0.994 / 0.995 | 0.979 / 0.974 / 0.985 |
| 0.7 | 0.963 / 0.972 / 0.955 | 0.936 / 0.978 / 0.896 | 0.995 / 0.998 / 0.992 | 0.983 / 0.992 / 0.974 |

The clean bimodal separation means the 0.5 threshold isn't delicate — slide toward 0.4 to catch
borderline misses (more false alarms) or toward 0.7 to cut false alarms (barely touching recall).

---

## 6. Placement robustness — defects anywhere

![position: centered vs offset](figures/g4_position.png)

The offset models are trained and tested with the defect placed randomly, so they hold up when a
real defect lands off-center — which the centered models do not. This is why the offset columns are
the deployment-relevant numbers. Example tiles with the defect at varying positions:

![placement examples](figures/examples_placement.png)

---

## 7. Resolution — and why 256 is *not* obsolete

![resolution: 256 vs 512](figures/g3_resolution.png)

On **pristine** pixels 512 wins everywhere (§2). But its advantage lives entirely in fine detail,
and fine detail is the first thing sensor noise destroys. Injecting read noise at inference
**flips the ordering**:

| read noise σ | 256 ROC-AUC | 512 ROC-AUC |
|---|---|---|
| 0 (pristine) | 0.972 | **0.996** |
| 5 | 0.967 | **0.975** |
| **10** | **0.938** | 0.811 |
| 20 | **0.835** | 0.655 |

At σ≈10 gray-levels, **256 becomes the more robust model** — its 2× downsample averages noise away,
while 512 feeds the noise straight into the classifier. So the resolution choice is conditional on
the imaging rig: **σ<5 → use 512; noisy sensor (σ≥10) → 256 is safer.** Full sweep across all five
nuisances (noise, exposure, blur, shift, rotation):

![nuisance sweep](figures/nuisance_sweep.png)

---

## 8. A note on the split (honest, in-distribution)

The split is per-(photo, defect) **unit** holdout — all 10 board designs appear in every split, so
this measures performance on **boards the model has seen**, matching the fixed-production-line
deployment plan. The earlier leak (a `SEED=42` crop-center collision that let the model memorize
its own test set) is gone: these datasets were freshly mined and dHash-checked. For a
*generalization* number (unseen board layouts), use the `--split-mode template` holdout instead
(~0.88).

---

## Artifacts & reproduce

- weights: `runs_resnet_v3/pcb_bin_{center,offset}_{256,512}/best.weights.h5`
- datasets: `datasets/pcb_bin_center`, `datasets/pcb_bin_offset`
- eval dumps: `resnet/details/eval_v3_*.txt`, `manifest_pcb_bin_*.json`,
  `confidence_v3.json`, `nuisance_{256,512}.json`

```bash
python resnet/eval_resnet.py --weights runs_resnet_v3/pcb_bin_offset_512/best.weights.h5 \
    --data datasets/pcb_bin_offset --size 512
python resnet/nuisance_sweep.py --weights <weights> --data <data> --size <256|512>
```

Companion docs: [`CONFIDENCE_REPORT.md`](CONFIDENCE_REPORT.md) (score distributions),
[`MODEL_REPORT_7CLASS.md`](MODEL_REPORT_7CLASS.md) (defect-type namer),
[`RESOLUTION_REPORT.md`](RESOLUTION_REPORT.md) / [`POSITION_REPORT.md`](POSITION_REPORT.md).
