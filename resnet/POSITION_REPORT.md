# Position report — does defect position affect detection? (Goal 4)

**TL;DR.** The centered-defect classifier only "sees" a defect near the tile center —
confidence collapses from **0.93 at center to 0.22 at 40% off-center** (catching 2/10).
Re-training with defects placed off-center (`--defect-offset 0.3`) **flattens the curve**: it
holds **≥0.82 out to 30% and catches 10/10 through 30%** off-center. Cost: overall accuracy
drops ~0.94 → 0.82 (position-invariance is a harder problem). Net: use the position-augmented
model for a sliding window so a defect landing anywhere in a tile still gets caught.

## Setup (controlled: only defect placement changes; in-distribution split)
- **before** = center-trained 256 model (`runs_resnet_at256/pcb_patches_512`, `--defect-offset 0`).
- **after** = same pipeline/size/split, BAD patches shifted off-center up to `0.3×patch`
  (`runs_resnet/pcb_patches_offcenter`, `--defect-offset 0.3`).
- Both on the **defect-level (in-distribution)** split. Position curve from `offcenter_test.py
  --source hr_08`: crop the source board so each defect sits at a growing offset from tile
  center, score P(defective), average over N=10 hr_08 defects. Both models trained on hr_08, so
  the comparison isolates centered-vs-offset training.

## P(defective) vs offset from center

```
offset(% of patch)   BEFORE (center-trained)      AFTER (offset-0.3)
                     meanP   caught>=0.5           meanP   caught>=0.5
   0%                0.925    9/10                 0.933   10/10
  10%                0.568    6/10                 0.917   10/10
  20%                0.730    9/10                 0.871   10/10
  30%                0.354    2/10                 0.818   10/10
  40%                0.221    2/10                 0.699    7/10
```

Plots: `resnet/offcenter_before.png`, `resnet/offcenter_after.png`.
(The "before" row at 10/20% is noisy — N=10 — but the trend is unmistakable: high at center,
collapsing by 30–40%.)

## Overall accuracy cost (in-distribution, each on its own test split)
| model | accuracy | precision | recall | ROC-AUC |
|---|---|---|---|---|
| center-trained (256) | **0.939** | 0.923 | 0.965 | 0.985 |
| position-augmented | 0.819 | 0.759 | **0.964** | 0.964 |

The position-augmented model keeps **defect recall high (0.964)** but its precision drops
(0.759) — it flags more clean patches — because "defect could be anywhere, even at the edge"
is a genuinely harder decision. Recall (catching defects) is what matters most for a screening
gate, and that holds up.

## Reading the result
- **Center bias is real and large.** The center model loses most of its confidence by 30–40%
  off-center and drops 7–8/10 defects below the 0.5 line.
- **Position augmentation removes it.** The "after" curve is flat to 20%, still catches 10/10
  at 30%, and only fades near 40% (defect approaching / leaving the tile edge).
- **The trade is peak/overall accuracy for coverage.** You give up ~12 points of in-distribution
  accuracy (mostly precision) to make the model catch defects anywhere in the tile.

## Deployment implication
- Use the **position-augmented model** for the sliding window: it lets tiles overlap *less*
  (coarser stride) while still catching edge-straddling defects — a board-level compute win.
- Keep any defect within ~30% of some tile's center (a stride ≲ 0.4·tile) and it is caught
  reliably (10/10). Beyond ~40% offset, rely on tile overlap.

## Caveats
- N=10 (hr_08 defects with in-bounds room); the effect is large but per-point meanP is a
  small-sample estimate — a `--source hr` sweep would tighten it.
- Measured on a training template (fair for isolating the centering intervention).
- Only `--defect-offset 0.3` tried; larger offsets extend the flat region at more accuracy cost.

## Artifacts (in-distribution)
- after: `runs_resnet/pcb_patches_offcenter/` → `runs_resnet_pcb_patches_offcenter_indist.zip`
- before: `runs_resnet_at256/pcb_patches_512/` → `runs_resnet_at256_pcb_patches_512_indist.zip`
- dataset: `datasets/pcb_patches_offcenter/` → `datasets_pcb_patches_offcenter_indist.zip`
- curves: `resnet/offcenter_before.png`, `resnet/offcenter_after.png`
- Reproduce: `python resnet/offcenter_test.py --weights <run>/best.weights.h5 --size 256 --source hr_08 --out <png>`
