# 7-class classifier — {good + 6 defect types}, defects anywhere

One model that both decides good/defective **and** names the defect, trained with defects placed
randomly anywhere in the tile (`--defect-offset 0.4`). The `good` row/column of the confusion
matrix *is* the good-vs-defective result.

**TL;DR.** top-1 **0.695**, macro-F1 **0.768**. It names distinctive defects very well
(`missing_hole` F1 0.985, `short` 0.915, `mouse_bite` 0.898) — but as a **good/bad gate it is
unusable**: it catches 99.5% of defects while **false-flagging 58% of good patches**.
**Keep good/bad and defect-type as two separate stages.**

## Setup
`datasets/pcb_types7_offset` (`mine_defect_types.py --split-mode defect --defect-offset 0.4
--good-per-plate 250`): one patch per annotated defect placed anywhere in the tile, plus a
`good` class of clean tiles from the healed plates. Per-(photo, defect) unit split
(in-distribution). Test = 2585 patches (1110 good + 1475 defects). ResNet-50, 7-way softmax,
two-phase, input 256. **Training wall-time ≈ 16 min** (GPU, two-phase; same order as the 256
binary models, ~half the 512 binary).

![The 7 classes](figures/examples_types.png)

## Results

overall top-1 **0.695** · macro-F1 **0.768**

![Confusion matrix](figures/g2b_confusion.png)

Per-class P / R / F1:
```
  missing_hole     1.000 / 0.970 / 0.985   (support 270)
  short            0.871 / 0.963 / 0.915   (support 190)
  mouse_bite       0.931 / 0.868 / 0.898   (support 310)
  open_circuit     0.865 / 0.837 / 0.851   (support 215)
  spur             0.574 / 0.862 / 0.689   (support 225)
  good             0.983 / 0.423 / 0.592   (support 1110)
  spurious_copper  0.299 / 0.902 / 0.449   (support 265)
```

## As a good/bad gate — collapse the 7×7

![Gate + per-class F1](figures/g2b_summary.png)

| | value |
|---|---|
| defect recall (bad caught) | **0.995** (1467/1475) |
| good specificity (clean passes) | **0.423** (470/1110) |
| false-alarm rate on good | **0.577** |
| overall good/bad accuracy | 0.749 |

The **dedicated binary gate**, on the same data and placement, reaches **0.984 accuracy / 0.985
recall** at 512 (0.997 precision at a higher threshold) — see [`MODEL_REPORT.md`](MODEL_REPORT.md).
The 7-class head throws that away.

### Threshold to minimize false negatives (call GOOD only if P(good) ≥ t)

Since a bad board slipping through is the costly error, we can raise the bar to earn a "good" call.
The gate is already very FN-averse — but driving misses to **zero** collapses specificity
(`details/gate7_threshold.json`):

| P(good) ≥ t | defect recall | FN (of 1475) | false-alarm | good passed |
|---|---|---|---|---|
| 0.0 (argmax) | 0.9959 | 6 | 57.8% | 468/1110 |
| 0.5 | 0.9966 | 5 | 64.4% | 395/1110 |
| **0.90** | **1.000** | **0** | 92.3% | 85/1110 |
| 0.99 | 1.000 | 0 | 99.7% | 3/1110 |

Zero false negatives is reachable at t ≥ 0.9, but at that point the gate flags **92%** of clean
boards — it is calling almost everything bad. So even at its most FN-averse, the 7-class gate can't
do the good/bad job; **use the binary 512 model as the gate** (which reaches recall 0.99 at 6.7%
false alarms) and keep this head for Stage-2 naming only.

## Why the good/bad boundary blurs
- **`good → spurious_copper` (507) and `good → spur` (101)** — clean copper traces read as small
  copper defects. Together they account for 55% of all good patches. That single confusion is
  what destroys the gate, and it also craters `spurious_copper`'s precision (0.299).
- **Defects-anywhere makes many BAD tiles mostly-clean** (defect at the edge), so the model
  becomes suspicious of genuinely clean tiles.
- **Inverse-frequency class weighting down-weights `good`** (the largest class), nudging the
  model to predict it less often — good recall 0.423 despite precision 0.983.

Notably `missing_hole` is near-perfect (F1 0.985) and no longer confused with `good`: with a
proper split the model learns real missing-holes cleanly, so normal drilled holes stop fooling it.
The residual difficulty is entirely the copper-texture family.

## Recommendation
```
Stage 1: binary good/bad gate  (512 input, position-augmented)  -> 0.984 acc / 0.985 recall
Stage 2: defect-type namer, only on tiles Stage 1 flagged bad
```
- Do **not** fold `good` into the type head — it costs ~0.57 in specificity.
- Trust `missing_hole` / `short` / `mouse_bite` / `open_circuit` calls; treat
  `spur` / `spurious_copper` as "copper anomaly, subtype uncertain."
- To fix the copper family: give the head more trace context (larger crop), or merge them into
  one `copper_anomaly` class if the application only needs actionable distinctions.

## Artifacts
- weights + manifest: `runs_resnet/pcb_types7_offset/` → `runs_resnet_pcb_types7_offset_v2.zip`
- dataset: `datasets/pcb_types7_offset/` → `datasets_pcb_types7_offset_v2.zip`
- classes (sorted): good, missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper
- Re-test: `python resnet/eval_multiclass.py --weights runs_resnet/pcb_types7_offset/best.weights.h5 --data datasets/pcb_types7_offset --size 256`
