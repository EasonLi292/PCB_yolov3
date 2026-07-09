# Detailed results — full record

Everything needed to audit or reproduce the headline numbers. Raw files live in
[`details/`](details/): per-model confusion matrices + threshold sweeps, run manifests, dataset
manifests, and the nuisance-sweep JSON.

---

## 1. Binary good/bad — the 2×2 (resolution × defect placement)

All four models trained with an **identical budget**: 30 frozen epochs (lr 1e-3) + 25 unfrozen
(lr 1e-5), batch 64 @256 / 16 @512 (largest that fits). Split = per-(photo, defect) unit
holdout, in-distribution. Test = 2294 patches (1110 good, 1184 bad); the good set is identical
across both datasets.

| model | acc @0.5 | acc @best | best thr | precision | recall | ROC-AUC | best val-AUC |
|---|---|---|---|---|---|---|---|
| 256 · centered | 0.957 | 0.963 | 0.7 | 0.950 | 0.969 | 0.995 | 0.9869 |
| 256 · offset 0.4 | 0.912 | 0.936 | 0.7 | 0.899 | 0.933 | 0.971 | 0.9590 |
| 512 · centered | 0.994 | 0.996 | 0.6 | 0.994 | 0.995 | 1.000 | 0.9982 |
| 512 · offset 0.4 | 0.979 | 0.984 | 0.8 | 0.974 | 0.985 | 0.996 | 0.9963 |

- **Cost of realistic (off-center) placement:** 256 −0.027, 512 −0.012 (best-threshold).
- **512-vs-256 gap:** +0.033 centered, +0.048 offset.
- **Every model's best threshold is above 0.5.** Reading accuracy at 0.5 costs 0.005–0.024.

Full confusion matrices and per-threshold sweeps: `details/eval_v3_*.txt`.

### Effect of training budget (why the first numbers were wrong)
| model | v2 (batch 32, 20+15 ep) | v3 (batch 64, 30+25 ep) |
|---|---|---|
| 256 centered | 0.935 | **0.957** |
| 256 offset | **0.820** | **0.912** |
| 512 centered | 0.994 | 0.994 |
| 512 offset | 0.989 | 0.979 |

The v2 256 models were under-trained. The published claim that "offset costs 256 −0.115
accuracy" was ~4× overstated: the true cost is −0.027.

---

## 2. Nuisance robustness — how much survives imperfect imaging

HRIPCB provides exactly **one photograph per board design** (verified: pixel-identical outside
the defect boxes), so its test set has **zero** board-to-board variation. `nuisance_sweep.py`
injects realistic rig imperfections at inference and re-measures. Both models are the
offset-trained (deployment) variants.

![Nuisance robustness](figures/nuisance_sweep.png)

**ROC-AUC** (threshold-independent):

| nuisance | magnitude | 256 | 512 |
|---|---|---|---|
| — | baseline | 0.972 | **0.996** |
| sensor noise | σ = 5 | 0.967 | 0.976 |
| sensor noise | σ = 10 | **0.938** | 0.811 |
| sensor noise | σ = 15 | **0.891** | 0.700 |
| sensor noise | σ = 20 | **0.835** | 0.655 |
| exposure drift | ±30% | 0.966 | 0.994 |
| focus blur | σ = 2.0 px | 0.955 | 0.973 |
| registration shift | 8 px | 0.969 | 0.995 |
| fixture rotation | 4° | 0.970 | 0.997 |

### What this says
- **Sensor noise is the only nuisance that flips the resolution decision.** 512 leads on clean
  pixels, but its AUC collapses (0.996 → 0.811 at σ=10) while 256 degrades gracefully
  (0.972 → 0.938). **Crossover at σ ≈ 5–10 gray levels.** Physically: 256's 2× downsample
  averages four pixels into one, halving noise; 512 keeps the high-frequency detail — and the
  noise riding on it. 512's advantage is an advantage *on pristine imagery*.
- **Focus blur** hurts 512 more (−0.023 AUC vs −0.017) but does **not** flip the ordering — 512
  stays ahead at every blur level tested.
- **Geometry is a non-issue.** An 8 px registration shift costs 256 only 0.009 accuracy and 512
  0.008; 4° of rotation costs ~0.019 / 0.018. ResNet-50's global average pooling delivers the
  translation invariance you'd expect. **Position sensitivity was never the problem.**
- **Exposure drift** of ±30% is essentially free for both.

### Rig specification implied
| parameter | requirement |
|---|---|
| sensor read noise | **σ < 5 gray levels** (mandatory if deploying 512; 256 tolerates σ ≈ 10) |
| focus | σ < 1 px blur |
| registration | ±8 px is fine — no tight fixturing needed |
| rotation | ±4° is fine |
| exposure stability | ±30% is fine |

---

## 3. Diagnostics — hypotheses tested and rejected

**(a) Aliasing in the data loader — REJECTED.**
`data.py` calls `tf.image.resize(img, (size,size))` with `antialias=False`, which looks like a
bug for the 512→256 downsample. Measured on a real patch:
```
mean|default_resize − cv2.INTER_AREA|   = 0.000   (identical)
mean|antialias=True − cv2.INTER_AREA|   = 0.846   (a different, blurrier filter)
```
At an *exact* 2× ratio TF's bilinear coincides with a box/area average — the correct operation.
No bug. **No code change made.** Recorded so this is not "fixed" later.

**(b) "Our 256 is under-trained because the release model gets 0.990 on our test set" — REJECTED
as evidence.**
The release model *did* score 0.990 / AUC 1.000 on `pcb_bin_center/test` — but it **memorized
it**. Both datasets were mined with `SEED=42`, so the random crop centers coincide. dHash
(64-bit) between our test patches and the release model's training pool:

| | exact (H=0) | H ≤ 2 | H ≤ 5 | median nearest |
|---|---|---|---|---|
| good | 27.9% | 78.9% | 98.5% | **1 bit** |
| bad | 26.2% | 77.3% | 95.9% | **1 bit** |

Validated that dHash discriminates these patches: **within** our own test split, 99.9% of hashes
are unique and the median nearest-neighbour distance is **10–12 bits**. A 1-bit median against
the release pool is genuine near-duplication. The release model is **not a valid reference**.

*(The v2→v3 retrain did still improve 256, so under-training was real — just not provable from
the release model.)*

**(c) "Accuracy collapse under offset" — RE-EXPLAINED as calibration.**
At 256 the offset model's accuracy fell 0.957 → 0.912 at threshold 0.5, but ROC-AUC only
0.995 → 0.971. Trained on BAD tiles that are mostly-clean-with-a-defect-near-the-edge, it learns
to fire on weaker evidence and sits over-confident toward "bad". At its best threshold (0.7) it
recovers to 0.936. **Always tune the threshold on a validation split.**

---

## 4. Split methodology and its assumptions

**Assumption: strict environment, repeatable imaging.** On a fixed line the design, camera and
fixture are constant, and systematic faults (worn drill, bad stencil aperture) recur at the same
spot. The same board and the same error *context* therefore appear in training and in
production. We do **not** hold out board designs.

**`--split-mode defect`** = per-(photo, defect) unit holdout:
- a *unit* = one defect box on one board image, or one good tile on a clean plate;
- all augmentation variants of a unit share a split (**no variant leak**);
- the same designs and photos appear in train / val / test (**in-distribution**).

2,953 bad units, split 80.7 / 9.3 / 10.0 — matching the granularity of the original
`split_manifest.csv`. *(An earlier key of `(template, defect-index)` ignored which photo a defect
came from: `tpl_04` had 3 units for 120 photos. Fixed.)*

**`--split-mode template`** (hold out whole layouts) answers a different question — "a product
design never seen before" — and scores ~0.88. Quote that for a *new* product, not a fixed line.

**Hard limit of this dataset.** HRIPCB has one photograph per design; every "board" of a template
is that photo with defects pasted in. There is no second physical board to hold out, so
board-to-board variation cannot be measured by *any* split. The nuisance sweep (§2) is the
substitute, and it is why the noise result matters so much.

---

## 5. Files

| file | contents |
|---|---|
| `details/eval_v3_{256,512}_{centered,offset}.txt` | confusion matrix + threshold sweep per model |
| `details/manifest_*.json` | run config, git commit, best val metric, re-test command |
| `details/dataset_manifest_*.json` | mining params, seed, split mode, counts |
| `details/nuisance_{256,512}.json` | full sweep results, machine-readable |
| `details/nuisance_sweep_both.txt` | sweep console output |
| `resnet/nuisance_sweep.py` | the sweep tool (inference only) |

Reproduce any model: `python resnet/eval_resnet.py --weights runs_resnet_v3/<run>/best.weights.h5 --data datasets/<ds> --size <256|512>`
Reproduce any dataset: re-run the `argv` recorded in its `dataset_manifest.json` (seeded, deterministic).
