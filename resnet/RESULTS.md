# PCB defect classifier — results

A ResNet-50 good/bad + defect-type classifier mined from the HRIPCB board set, evaluated for a
**fixed production line with strict, repeatable imaging**: the same board designs the model
trained on, with defects landing anywhere a sliding window happens to find them.

Full audit trail — confusion matrices, threshold sweeps, manifests, diagnostics:
**[DETAILED_RESULTS.md](DETAILED_RESULTS.md)**.

## How the data is made

A defective board is ~99% defect-free *area*, so we mine patches. **BAD** = crops around an
annotated defect; **GOOD** = crops of a *reconstructed clean plate* (the per-pixel median of a
template's aligned copies washes the defects out). Good and bad come from the **same boards at
the same zoom**, so the model can only cheat on the defect itself.

![Defective board vs reconstructed clean plate](figures/examples_board.png)

Defects are placed **anywhere in the tile**, not conveniently centered — what a sliding window
actually sees:

![Defect placement](figures/examples_placement.png)

---

## 1 · Resolution — and the noise floor that decides it

With an **equal training budget**, 512 leads 256 by a modest margin: **+0.033** accuracy on
centered defects, **+0.048** on realistic off-center ones. Off-center placement costs 256 only
**−0.027**, not the collapse an earlier under-trained run suggested.

![Resolution x placement](figures/g3_resolution.png)

| | acc @best thr | ROC-AUC |
|---|---|---|
| 256 · centered | 0.963 | 0.995 |
| 256 · offset | 0.936 | 0.971 |
| 512 · centered | 0.996 | 1.000 |
| 512 · offset | 0.984 | 0.996 |

**But that lead exists only on pristine pixels.** Inject realistic sensor noise and the ordering
**flips** — 256's 2× downsample averages noise away, while 512 keeps the high-frequency detail
*and* the noise on it:

![Nuisance robustness](figures/nuisance_sweep.png)

| sensor noise σ | 256 AUC | 512 AUC |
|---|---|---|
| 0 | 0.972 | **0.996** |
| 5 | 0.967 | 0.976 |
| 10 | **0.938** | 0.811 |
| 20 | **0.835** | 0.655 |

**Crossover at σ ≈ 5–10 gray levels.** Registration shift (8 px), rotation (4°) and exposure
drift (±30%) are near-free for both models.

→ **Measure your rig's read noise before committing to 512.** [RESOLUTION_REPORT.md](RESOLUTION_REPORT.md)

---

## 2 · Position — a training artifact, not an architecture limit

A centered-trained model goes blind off-center (mean P falls 0.86 → 0.02 at 40% offset; **0/10**
caught). Training with defects anywhere fixes most of it (9–10/10 through 30%), and costs only
0.027 accuracy at 256 / 0.012 at 512.

![Position curves](figures/g4_position.png)

This is **not** a failure of translation invariance — shifting the whole tile 8 px costs 0.009
accuracy. What changes off-center is the surrounding context and proximity to the zero-padded
border. → [POSITION_REPORT.md](POSITION_REPORT.md)

---

## 3 · One 7-class model — good + defect type

Names distinctive defects well (`missing_hole` F1 0.985, `short` 0.915, `mouse_bite` 0.898), but
as a good/bad gate it is unusable: catches 99.5% of defects while **false-flagging 58% of good**
patches — clean copper traces read as `spurious_copper`.

![7-class confusion](figures/g2b_confusion.png)

→ **Keep two stages.** [MODEL_REPORT_7CLASS.md](MODEL_REPORT_7CLASS.md)

---

## Recommended deployment

```
board → sliding window (stride so any defect lands within ~30% of a tile center)
      → Stage 1: binary good/bad gate  (offset-trained; threshold ≈ 0.7, NOT 0.5)
            resolution: 512 if sensor read noise σ < 5 gray levels, else 256
            good → pass
            bad  → Stage 2: defect-type namer → report the defect
```

Rig spec implied by the sweep: **read noise σ < 5** (the binding constraint), focus σ < 1 px,
registration ±8 px, rotation ±4°, exposure ±30%.

---

## Methodology and its assumptions

**Assumption: strict environment, repeatable imaging.** On a fixed line the design, camera and
fixture are constant, and systematic faults recur at the same spot — so the same board and error
*context* legitimately appear in training and production. We therefore do **not** hold out board
designs.

**The split** (`--split-mode defect`) is a per-(photo, defect) **unit holdout**: a unit is one
defect box on one board image (or one good tile on a clean plate); all augmentation variants of a
unit share a split (no variant leak); the same designs and photos appear in train/val/test.
2,953 bad units, split 80.7 / 9.3 / 10.0 — matching the original `split_manifest.csv`.

**Hard limit of this dataset.** HRIPCB has exactly **one photograph per board design** (verified:
pixel-identical outside the defect boxes). There is no second physical board to hold out, so
board-to-board variation cannot be measured by *any* split. The nuisance sweep is the substitute —
and it is why the noise result above matters more than the clean-data table.

**`--split-mode template`** (hold out whole layouts) answers a different question — "a product
design never seen before" — and scores ~0.88. Quote that for a *new* product, not a fixed line.

## Corrections made
Earlier versions of these reports overstated two effects. Both are documented in
[DETAILED_RESULTS.md](DETAILED_RESULTS.md) §3:
1. **"Offset costs 256 −0.115 accuracy"** → the 256 models were under-trained *and* read at the
   wrong threshold. True cost: **−0.027**.
2. **"The release 256 model proves ours is under-trained"** → the release model **memorized** our
   test set (same `SEED=42` crop centers; 28% exact dHash matches, median nearest distance 1 bit).
   Not a valid reference.
3. A suspected **aliasing bug** in the data loader was tested and **rejected** (TF bilinear ==
   `INTER_AREA` at an exact 2× ratio). No code change.

## Saved artifacts
Weights `runs_resnet_v3/pcb_bin_{center,offset}_{256,512}/` · datasets `datasets/pcb_bin_{center,offset}/`
· 7-class `runs_resnet/pcb_types7_offset/`. Every run has `run_manifest.json`, every dataset a
seeded `dataset_manifest.json`. Raw evals and sweeps in [`details/`](details/).
