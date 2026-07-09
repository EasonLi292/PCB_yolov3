# Position report — does an off-center defect still get caught?

**TL;DR.** A centered-trained classifier has a **center blind spot**: confidence falls from
0.91 at the tile center to 0.33 at 40% off-center, catching only **3/10** defects. Training
with defects placed anywhere (`--defect-offset 0.4`) **eliminates it**: the model catches
**10/10 defects at every offset out to 40%**, and its response stays ≥0.77 throughout.

## Why it matters
At inference the classifier slides over the board in tiles. A defect lands wherever the grid
puts it — rarely dead-center. If the model only fires on centered defects, you must overlap
tiles heavily (expensive) to guarantee coverage.

![Defect placement](figures/examples_placement.png)

## Setup (controlled — only defect placement differs)
- **before** = 256 model trained on centered defects (`runs_resnet_at256/pcb_bin_center`)
- **after** = same size/recipe/split, defects placed anywhere (`runs_resnet_at256/pcb_bin_offset`,
  `--defect-offset 0.4`)
- Curve from `offcenter_test.py --source hr_08`: crop the source board so a real annotated
  defect sits at a growing offset from tile center, score P(defective), average over N=10
  defects. Both models trained on hr_08, so the comparison isolates the centering intervention.

## P(defective) vs offset

![Position curves](figures/g4_position.png)

```
offset(% of tile)   BEFORE (centered)        AFTER (offset-0.4)
                    meanP  caught>=0.5       meanP  caught>=0.5
   0%               0.910    9/10            0.914   10/10
  10%               0.679    8/10            0.916   10/10
  20%               0.683    8/10            0.959   10/10
  30%               0.371    3/10            0.887   10/10
  40%               0.331    3/10            0.769   10/10
```

The "after" model never drops a defect — 10/10 at every offset, including 40%, where the
defect is nearly at the tile edge. (Training at offset 0.4 also beats the earlier 0.3-trained
model, which faded to 7/10 at 40%.)

## The cost, and where it lands
Position invariance is not free — it makes the task harder, and **the price depends entirely on
resolution**:

| | accuracy centered → offset | precision centered → offset |
|---|---|---|
| 256 | 0.935 → **0.820** | 0.903 → **0.755** |
| 512 | 0.994 → **0.989** | 0.990 → **0.997** |

At **512 the cost is essentially zero** (−0.005 accuracy, precision actually improves). At 256
it is severe: the model, taught that a mostly-clean tile can still be bad, floods good patches
with false alarms. **So position augmentation and 512 input go together** — see
[RESOLUTION_REPORT.md](RESOLUTION_REPORT.md).

## Deployment implication
- Train with off-center defects **and** run at 512. You then get a flat position response with
  no accuracy penalty, which permits a **coarser sliding stride** (fewer tiles per board) — a
  net compute win at the board level.
- Set the stride so any defect lands within ~40% of some tile's center; the curve says that band
  is caught 10/10.

## Caveats
- N = 10 (hr_08 defects with enough in-bounds room at patch=1024); the effect is large and
  monotonic, but each point is a small-sample mean. A `--source hr` sweep would tighten it.
- Measured on a training template — this measures the *capability* to respond off-center, not
  generalization to an unseen layout.
- Offset 0.4 keeps the defect fully framed. Match it to your real window geometry.

## Artifacts
- after: `runs_resnet_at256/pcb_bin_offset/` → `runs_resnet_at256_pcb_bin_offset_v2.zip`
- before: `runs_resnet_at256/pcb_bin_center/` → `runs_resnet_at256_pcb_bin_center_v2.zip`
- curves: `resnet/offcenter_before.png`, `resnet/offcenter_after.png`
- Reproduce: `python resnet/offcenter_test.py --weights <run>/best.weights.h5 --size 256 --source hr_08 --out <png>`
