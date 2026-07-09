# PCB defect classifier — results

A ResNet-50 good/bad + defect-type classifier mined from the HRIPCB board set, evaluated for a
**fixed production line with strict, repeatable imaging**: the same board designs the model
trained on, with defects landing anywhere a sliding window happens to find them.

## How the data is made

A defective board is ~99% defect-free *area*, so we mine patches. **BAD** = crops around an
annotated defect; **GOOD** = crops of a *reconstructed clean plate* (the per-pixel median of a
template's aligned copies washes the defects out). Good and bad come from the **same boards at
the same zoom**, so the model can only cheat on the defect itself.

![Defective board vs reconstructed clean plate](figures/examples_board.png)

Crucially, defects are placed **anywhere in the tile**, not conveniently centered — this is
what a sliding window actually sees:

![Defect placement](figures/examples_placement.png)

---

## 1 · Resolution × placement — the headline

**512 is what makes the classifier work.** Under realistic (off-center) defects, 512 barely
notices the offset while 256 collapses — its precision falls to 0.755, flooding good boards
with false alarms.

![Resolution x placement](figures/g3_resolution.png)

| | accuracy | precision | recall | ROC-AUC |
|---|---|---|---|---|
| 256 · centered | 0.935 | 0.903 | 0.979 | 0.992 |
| 256 · **offset** | **0.820** | 0.755 | 0.965 | 0.967 |
| 512 · centered | 0.994 | 0.990 | 0.999 | 1.000 |
| 512 · **offset** | **0.989** | **0.997** | 0.981 | 0.997 |

Realistic placement widens the 512-vs-256 gap from **+0.06 to +0.17 accuracy**. A centered-only
comparison understates the case for 512. → [RESOLUTION_REPORT.md](RESOLUTION_REPORT.md)

---

## 2 · Position — the center blind spot, and its cure

A centered-trained model stops seeing a defect once it leaves the middle of the tile (3/10
caught at 30% off-center). Training with defects anywhere fixes it completely: **10/10 caught at
every offset out to 40%.** At 512 this costs essentially nothing (−0.005 accuracy).

![Position curves](figures/g4_position.png)

→ [POSITION_REPORT.md](POSITION_REPORT.md)

---

## 3 · One 7-class model — good + defect type

It names distinctive defects well (`missing_hole` F1 0.985, `short` 0.915, `mouse_bite` 0.898),
but as a good/bad gate it is unusable: it catches 99.5% of defects while **false-flagging 58% of
good patches** — clean copper traces read as `spurious_copper`.

![7-class confusion](figures/g2b_confusion.png)

![7-class gate](figures/g2b_summary.png)

→ **Keep two stages.** [MODEL_REPORT_7CLASS.md](MODEL_REPORT_7CLASS.md)

---

## Recommended deployment

```
board → sliding window, 512×512 tiles (coarse stride is fine — position-invariant)
      → Stage 1: binary good/bad gate   (512, position-augmented)   0.989 acc / 0.997 precision
            good → pass
            bad  → Stage 2: defect-type namer  → report the defect
```

---

## Methodology — how the split works, and what it assumes

**Assumption: strict environment, repeatable imaging.** On a fixed line the board design, camera,
and fixture are constant, and systematic faults (a worn drill, a bad stencil aperture) recur at
the same spot. So the same board and the same error *context* legitimately appear in training and
in production. We therefore do **not** hold out board designs.

**The split** (`--split-mode defect`) is a **per-(photo, defect) unit holdout**:
- a *unit* = one defect box on one board image, or one good tile on a clean plate;
- all augmentation variants of a unit share a split (**no variant leak**);
- the same board designs and photos appear in train, val and test (**in-distribution**).

This reproduces the granularity of the original `split_manifest.csv`: 2,953 bad units, split
80.7 / 9.3 / 10.0. *(An earlier version keyed units on `(template, defect-index)`, which ignored
which photo a defect came from — `tpl_04` had 3 units for 120 photos. Fixed.)*

**Known limits of this dataset.** HRIPCB has exactly **one photograph per board design**; every
"board" of a template is that same photo with defects pasted in (verified: pixel-identical
outside the defect boxes). So there is **zero board-to-board nuisance variation** — no sensor
noise, lighting drift, or registration jitter. These numbers therefore assume *perfect* imaging
repeatability. A real rig has some variation; expect a modest haircut. The honest way to bound
that is a nuisance-robustness sweep (inject noise / gain / sub-pixel shift at test time), not a
different split — there is no second physical board to hold out.

The alternative, **`--split-mode template`** (hold out whole board layouts), answers a different
question — "how does it do on a product design it has never seen?" — and scores ~0.88. That is the
number to quote for a *new* product, not for a fixed line.

## Saved artifacts (reproducible without retraining)

Every run has `run_manifest.json` (config + git commit + exact re-test command); every dataset a
`dataset_manifest.json` (seeded → regenerable).

| experiment | weights | dataset |
|---|---|---|
| 512 / 256 · offset | `runs_resnet_pcb_bin_offset_v2.zip` · `runs_resnet_at256_pcb_bin_offset_v2.zip` | `datasets_pcb_bin_offset_v2.zip` |
| 512 / 256 · centered | `runs_resnet_pcb_bin_center_v2.zip` · `runs_resnet_at256_pcb_bin_center_v2.zip` | `datasets_pcb_bin_center_v2.zip` |
| 7-class | `runs_resnet_pcb_types7_offset_v2.zip` | `datasets_pcb_types7_offset_v2.zip` |

*(A defects-only 6-class variant and the YOLO detector comparison are not featured here.)*
