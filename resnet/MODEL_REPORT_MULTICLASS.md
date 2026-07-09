# Defect-type classifier report — 6-class (Goal 2)

**TL;DR (in-distribution).** A ResNet-50 softmax head over the 6 HRIPCB defect classes reaches
**top-1 0.825 / macro-F1 0.829** when the board designs were seen in training. `missing_hole`
is **perfect (F1 1.000)**, `short` and `open_circuit` are strong (0.91, 0.89); the
copper-texture classes (`spur`, `spurious_copper`, `mouse_bite`) still trade among themselves.
(On unseen board layouts this falls to 0.60 — see the contrast below.) Use as the *second
stage* after the binary good/bad gate.

## Setup
- Data: `datasets/pcb_defect_types` (`mine_defect_types.py --split-mode defect`): one patch per
  annotated defect, labeled by class, **defect-level split** (same board designs in train &
  test). Test = 1990 patches. Model `build_resnet50(n_classes=6)`, two-phase, input 256.

## Results — IN-DISTRIBUTION
overall top-1 **0.825** · macro-F1 **0.829**

```
confusion (rows=true, cols=pred)
             missing  mouse_b  open_ci    short     spur  spuriou
missing_hole     365        0        0        0        0        0
mouse_bite         0      171       10        1       60        8
open_circuit       0       11      302       10       18        9
short              0        0        3      192        3        2
spur               0       56       12       17      429       91
spurious_copper    0        1        0        2       36      196
```

Per-class P / R / F1:
```
  missing_hole     1.000 / 1.000 / 1.000   (support 365)
  short            0.865 / 0.960 / 0.910   (support 200)
  open_circuit     0.924 / 0.863 / 0.892   (support 350)
  spur             0.786 / 0.709 / 0.745   (support 605)
  spurious_copper  0.641 / 0.834 / 0.725   (support 235)
  mouse_bite       0.715 / 0.684 / 0.699   (support 250)
```

## Reading it
- **`missing_hole` is perfectly separable** — a dark circular void is unlike any copper defect.
- **`short` and `open_circuit`** are well identified (F1 0.91 / 0.89).
- **The copper-texture family still mixes**, but far less than out-of-distribution: the residual
  confusion is `spur ↔ spurious_copper` (91 + 36) and `mouse_bite ↔ spur` (60 + 56) — small
  copper add/remove artifacts that look alike at patch scale.

## Contrast — held-out TEMPLATE split (unseen layouts)
Top-1 **0.596**, macro-F1 0.595; `missing_hole` drops to F1 0.88 and the copper family collapses
together. Naming a defect on a board design never seen is much harder; on a known design the
model reaches 0.83. Both are valid — pick the one matching your deployment.

## Recommendation
- **Two-stage pipeline:** binary good/bad gate decides *if* defective (its job), then this
  model names the type on flagged patches. Don't use it standalone.
- **Trust per-class:** act on `missing_hole` / `short` / `open_circuit`; treat a
  `spur`/`spurious_copper`/`mouse_bite` call as "copper anomaly (subtype uncertain)".
- To sharpen the copper subtypes: more trace context around the defect (larger crop), or the
  full-board detector (YOLO), which sees context a centered patch loses.

## Caveats
- Defect-level split (same board designs in train/test) — the realistic known-product number.
- Patch-scale typing discards board context that partly defines these classes.
- Defects-only dataset (assumes a defect is present — the binary gate's job).

## Artifacts
- weights + manifest: `runs_resnet/pcb_defect_types/` → `runs_resnet_pcb_defect_types_indist.zip`
- dataset: `datasets/pcb_defect_types/` → `datasets_pcb_defect_types_indist.zip`
- Re-test: `python resnet/eval_multiclass.py --weights runs_resnet/pcb_defect_types/best.weights.h5 --data datasets/pcb_defect_types --size 256`
