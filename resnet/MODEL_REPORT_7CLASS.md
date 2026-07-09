# 7-class classifier report — {good + 6 defect types}, defects anywhere (Goal 2b)

**What & why.** One model that both decides good/defective *and* names the defect, with defects
placed **randomly anywhere in the tile** (`--defect-offset 0.4`) to simulate a real board where
a sliding window catches a defect at a random position. The `good` row/column of the confusion
matrix is the good-vs-defective result.

**TL;DR (in-distribution).** top-1 **0.685**, macro-F1 **0.742**. As a defect *detector* it
catches **99% of defects** but **false-flags 54% of good patches** — poor specificity, even
in-distribution. Folding `good` into the type head is a bad screening gate. **Keep the two-stage
design: a dedicated binary good/bad gate → type namer.**

## Setup
`datasets/pcb_defect_types7` (`mine_defect_types.py --split-mode defect --defect-offset 0.4
--good-per-plate 250`): 6 defect classes (one patch per defect, placed anywhere) + a `good`
class (clean tiles from healed plates). Defect-level split (same designs in train & test).
Test = 3215 patches (1210 good + 2005 defects). ResNet-50, 7-way softmax, two-phase, input 256.

## Results — IN-DISTRIBUTION
overall top-1 **0.685** · macro-F1 **0.742**

```
confusion (rows=true, cols=pred)
                good  missing  mouse_b  open_ci    short     spur  spuriou
good             552        0        9        6        5      255      383
missing_hole       0      360        0        0        0        1        4
mouse_bite         1        0      178        6        3       47       15
open_circuit       0        0        2      290        1       36       21
short              0        0        0        1      191        6        2
spur              12        0       51        2        3      434      103
spurious_copper    3        0        0        3        0       31      198
```

Per-class P/R/F1: good 0.972/0.456/0.621 · missing_hole 1.000/0.986/0.993 · short 0.941/0.955/0.948 ·
open_circuit 0.942/0.829/0.881 · mouse_bite 0.742/0.712/0.727 · spur 0.536/0.717/0.613 ·
spurious_copper 0.273/0.843/0.412.

## Good-vs-defective (collapse the 7×7 to a detector)
| | value |
|---|---|
| defect recall (bad caught) | **0.992** (1989/2005) |
| good specificity (clean passes) | **0.456** (552/1210) |
| false-alarm rate on good | **0.544** |
| overall good/bad accuracy | 0.789 |

Compare the **dedicated binary gate** (in-distribution, same data): accuracy **0.94–0.99**,
good specificity **~0.9**. The single 7-class model throws away ~half of that specificity —
false-flagging half the good boards is unacceptable for a screening gate.

## Why the good/bad boundary blurs — even in-distribution
1. **`good → spur` (255) and `good → spurious_copper` (383).** Clean copper traces on a good
   board look like small copper defects; ~53% of good patches are called some copper defect.
   (Note the earlier held-out `good↔missing_hole` confusion is *gone* here — in-distribution the
   model learns real missing-holes perfectly (F1 0.99), so normal holes no longer fool it; the
   confusion simply moved to the copper family.)
2. **Defects-anywhere makes many BAD patches mostly-clean** (defect at a tile edge), so the model
   grows suspicious of genuinely clean patches too.
3. **Class weighting** down-weights `good` (the majority class), nudging the model to predict it
   less often → low good recall.

## Recommendation
- **Keep two stages** — dedicated binary good/bad gate (high specificity, ~0.9) → type namer.
  Do not fold `good` into the type model; it costs ~0.4 in specificity.
- If a single model is required, expect ~0.5 false alarms; fixing it means de-confusing `good`
  from the copper family (more trace context; don't sample `good` tiles over copper-dense areas).
- **Defects-anywhere is right for the detector** (it made the position-augmented binary model
  robust — `POSITION_REPORT.md`); it just makes fine typing harder. Type accuracy among defects
  here is ~0.72, consistent with the 6-class model under the same anywhere-placement.

## Contrast — held-out TEMPLATE split
top-1 0.532; good specificity 0.487 / defect recall 0.922; the dominant good confusion there was
`good↔missing_hole`. Same overall conclusion (poor gate), different confusion partner.

## Artifacts
- weights + manifest: `runs_resnet/pcb_defect_types7/` → `runs_resnet_pcb_defect_types7_indist.zip`
- dataset: `datasets/pcb_defect_types7/` → `datasets_pcb_defect_types7_indist.zip`
- classes (sorted): good, missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper
- Re-test: `python resnet/eval_multiclass.py --weights runs_resnet/pcb_defect_types7/best.weights.h5 --data datasets/pcb_defect_types7 --size 256`
