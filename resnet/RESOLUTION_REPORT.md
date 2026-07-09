# Resolution report — 512×512 vs 256×256 (Goal 3)

**TL;DR (in-distribution).** When the board designs in the test set were also seen in training
(the real production-line case), **512 clearly beats 256**: accuracy **0.992 vs 0.939**,
ROC-AUC **0.998 vs 0.985**. Higher input resolution preserves fine defect detail the model can
exploit on boards it knows. This *reverses* the held-out-template finding (below), where the
two tied — so the resolution verdict depends on which deployment you mean.

## Setup
Both models trained on the **same** dataset (`datasets/pcb_patches_512`, HRIPCB healed good/bad,
`--split-mode defect`), only the model input `--size` differs. **Split = defect-level**: the
same board *designs* appear in train and test (augmentation variants of one defect stay within
a split, so no exact-patch leak). Two-phase (frozen head → unfrozen fine-tune). Test = 2493
patches (1173 good, 1320 bad).

## Results — IN-DISTRIBUTION (same board designs in train & test)

| Metric (test @0.50) | 256-input | 512-input |
|---|---|---|
| accuracy | 0.939 | **0.992** |
| precision (defect) | 0.923 | **0.992** |
| recall (defect) | 0.965 | **0.994** |
| F1 | 0.943 | **0.993** |
| ROC-AUC | 0.985 | **0.998** |
| PR-AUC | 0.990 | **0.999** |
| relative compute (MACs ∝ H·W) | **1×** | 4× |

```
256-input                     512-input
           pred good  pred bad            pred good  pred bad
actual good     1066       107  actual good    1162        11
actual bad        46      1274  actual bad        8      1312
```

At 512 the classifier is essentially saturated (0.99 everywhere); at 256 it makes ~3× as many
mistakes (107 vs 11 false alarms on good, 46 vs 8 missed defects).

## The contrast that matters — held-out TEMPLATE split (unseen board layouts)
The same experiment where test uses **entirely unseen board designs**:

| Metric | 256-input | 512-input |
|---|---|---|
| accuracy | 0.880 | 0.734 |
| ROC-AUC | 0.933 | 0.928 |

Here 512 gives **no** discrimination gain (AUC tied) — the extra detail is board-specific and
does not transfer to novel layouts; it only shifts the operating point.

## Interpretation
- **On boards you trained on, resolution helps** (0.985 → 0.998 AUC): the model can lock onto
  fine, board-specific defect texture, and 512 keeps that texture the 256 downscale throws away.
- **On brand-new boards, resolution does not help** (AUC tied): the transferable signal is
  coarse, so 512 just costs 4× compute for nothing.

## Recommendation
- **Fixed product line (inspecting known board designs): 512 is worth it** — it takes the
  classifier from 0.94 to 0.99 accuracy. Budget the 4× DLA cost; re-export the IR at
  `--size 512` if deploying it.
- **Must generalize to unseen board designs: stay at 256** (or 384) — 512 buys no AUC there.
- 384 is a reasonable middle (≈2.25× 256 compute) if you want some of the fidelity gain at
  lower cost.

## Caveats
- Defect-level split shares board *designs* (not identical patches) across train/test — the
  realistic "same product" scenario, not exact-image leakage.
- HRIPCB healed (median) GOOD vs real defective BAD is a slightly easier separation than
  real-vs-real; absolute numbers on real clean boards would be a touch lower.

## Artifacts (in-distribution)
- 512-input: `runs_resnet/pcb_patches_512/` → `runs_resnet_pcb_patches_512_indist.zip`
- 256-input: `runs_resnet_at256/pcb_patches_512/` → `runs_resnet_at256_pcb_patches_512_indist.zip`
- dataset: `datasets/pcb_patches_512/` (+manifest, `split_mode: defect`) → `datasets_pcb_patches_512_indist.zip`
- Held-out-template versions of all three are in the non-`_indist` zips.
- Re-test: `python resnet/eval_resnet.py --weights <run>/best.weights.h5 --data datasets/pcb_patches_512 --size <256|512>`
