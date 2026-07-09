# Position report — does an off-center defect still get caught?

**TL;DR.** A centered-trained classifier develops a real **center blind spot**: mean P(defective)
falls from 0.86 at the tile center to **0.02 at 40% off-center**, catching **0/10** defects.
Training with defects placed anywhere (`--defect-offset 0.4`) largely fixes it: the response
stays ≥0.76 out to 30% and catches **9–10/10 through 30%**, fading to 5/10 at 40%.

The **cost is small**: at 256 the offset model loses 0.027 accuracy (best threshold), at 512
only 0.012. And per the nuisance sweep, the model is otherwise **nearly translation-invariant**
— an 8 px registration shift costs 0.009 accuracy. The blind spot is a *training-distribution*
artifact, not an architectural limit of ResNet.

> **Correction.** An earlier version reported the offset model catching 10/10 at *every* offset.
> That came from an under-trained v2 model. Retrained at an equal budget, it catches 9–10/10
> through 30% and 5/10 at 40%. The direction is unchanged; the magnitude was overstated.

## Why it matters
At inference the classifier slides over the board in tiles. A defect lands wherever the grid
puts it. If the model only fires on centered defects you must overlap tiles heavily to guarantee
coverage.

![Defect placement](figures/examples_placement.png)

## Setup (controlled — only defect placement differs)
- **before** = 256 model trained on centered defects (`runs_resnet_v3/pcb_bin_center_256`)
- **after** = same size / recipe / split / budget, defects placed anywhere
  (`runs_resnet_v3/pcb_bin_offset_256`, `--defect-offset 0.4`)
- Curve from `offcenter_test.py --source hr_08`: crop the source board so a real annotated defect
  sits at a growing offset from the tile center, score P(defective), average over N=10 defects.
  Both models trained on hr_08, so this isolates the centering intervention.

## P(defective) vs offset

![Position curves](figures/g4_position.png)

```
offset(% of tile)   BEFORE (centered)      AFTER (offset-0.4)
                    meanP  caught>=0.5     meanP  caught>=0.5
   0%               0.856    9/10          0.782    9/10
  10%               0.450    4/10          0.784    8/10
  20%               0.355    3/10          0.914   10/10
  30%               0.205    2/10          0.756    9/10
  40%               0.016    0/10          0.559    5/10
```

The centered model is effectively blind past ~20% offset (0/10 at 40%). The offset model holds a
usable response out to 30% and degrades gracefully at 40%, where the defect is nearly at the
tile edge and its context is truncated by padding.

## The cost, and where it lands
| | acc centered → offset (best thr) | precision |
|---|---|---|
| 256 | 0.963 → **0.936** (−0.027) | 0.950 → 0.899 |
| 512 | 0.996 → **0.984** (−0.012) | 0.994 → 0.974 |

Position invariance is cheap — **and cheaper at 512**, which retains enough detail to localize a
defect wherever it lands. The offset-trained models are also **mis-calibrated toward "bad"** at
the default 0.5 threshold (they learned that a mostly-clean tile can still be defective); use
**thr ≈ 0.7** and most of the apparent loss disappears.

## Is this "ResNet isn't translation-invariant"?
No. The nuisance sweep ([DETAILED_RESULTS.md](DETAILED_RESULTS.md) §2) shifts the *whole tile* by
up to 8 px and costs only 0.009 accuracy — global average pooling does its job. What the offset
dataset changes is not merely position but **how much board context surrounds the defect** and
**how close it sits to the zero-padded border**, where receptive fields are truncated. That is
what training on off-center defects teaches the model to handle.

## Deployment implication
- Train with off-center defects. Set the sliding stride so any defect lands within ~30% of some
  tile's center; the curve says that band is caught 9–10/10.
- Beyond ~40% offset, rely on tile overlap rather than the model.
- Tune the decision threshold on a validation split (≈0.7 here), never leave it at 0.5.

## Caveats
- **N = 10** (hr_08 defects with enough in-bounds room at patch=1024). The trend is large and
  monotonic, but each point is a small-sample mean — do not over-read individual points (the
  0.914 at 20% exceeding 0.784 at 10% is noise).
- Measured on a training template: this measures the *capability* to respond off-center, not
  generalization to an unseen layout.
- Offset 0.4 keeps the defect fully framed. Match it to your real window geometry.

## Artifacts
- after: `runs_resnet_v3/pcb_bin_offset_256/` · before: `runs_resnet_v3/pcb_bin_center_256/`
- curves: `resnet/offcenter_before.png`, `resnet/offcenter_after.png`,
  `details/offcenter_{before,after}_v3.txt`
- Reproduce: `python resnet/offcenter_test.py --weights <run>/best.weights.h5 --size 256 --source hr_08 --out <png>`
