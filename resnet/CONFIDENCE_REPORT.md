# ResNet-50 PCB Classifier — Confidence-Score Distribution

How the model's output `P(defective)` is distributed across the test set
(`datasets/pcb_patches/test`, 2,384 patches, FP32), color-coded by true class and by
whether the prediction is correct. Companion to [`MODEL_REPORT.md`](MODEL_REPORT.md).

**Color key:** 🟩 green = truly **good**, 🟥 red = truly **defective**. Dashed line = the
0.5 decision threshold. Good patches *should* sit near 0, defective near 1.

![confidence distribution](confidence_distribution.png)

## Reading the three panels

1. **Log histogram (top)** — shows the full shape. The two classes are almost completely
   separated and pile up at the extremes: good against 0, defective against 1. The shaded
   tints mark the "wrong side" of the threshold for each class.
2. **Linear, clipped to 60 (middle)** — the spikes at 0 and 1 run off-chart on purpose so
   the **contested middle** is legible. This is the only region where errors can occur;
   the arrows point at the 4 false negatives (defect scored < 0.5) and the 41 false
   positives (good scored ≥ 0.5).
3. **Per-patch strip (bottom)** — every test patch as one dot at its score, good on the
   lower row, defective on the upper. Correct predictions are faint dots; **misclassified
   patches are ringed and ×-marked** so you can spot exactly where and how many.

## What the distribution says

| | mean | median | piled at its correct extreme |
|---|---|---|---|
| **good** (🟩) | 0.059 | 0.005 | 86.5 % score < 0.1 |
| **defective** (🟥) | 0.994 | 1.000 | 98.5 % score > 0.9 |

- **The model is decisive, not hedging.** Almost every patch lands within 0.1 of 0 or 1;
  only **172 of 2,384 patches (7 %)** fall in the contested 0.1–0.9 band — and that band
  is **155 good vs. 17 defective**, i.e. the uncertainty is overwhelmingly on the good
  side (the model second-guesses clean patches far more than defective ones).
- **Errors live at the boundary, and they're lopsided the safe way.** All 45 mistakes
  (41 FP + 4 FN) sit near 0.5. Because false alarms (good→defect) outnumber misses
  (defect→good) **10 : 1**, the failure mode is *over-cautious* — the right direction for
  a screen whose expensive mistake is shipping a bad board.
- **The 4 missed defects barely cross the line.** Three score 0.35–0.49 (a lower
  threshold recovers them); only one is a confident miss at 0.066. See
  [`false_negatives.jpg`](false_negatives.jpg).
- **Well-conditioned for lower precision.** With scores this far from 0.5, FP16 moved only
  2 predictions ([`MODEL_REPORT_FP16.md`](MODEL_REPORT_FP16.md)) — the wide margin is why
  precision reduction is nearly free here.

**Practical takeaway:** the clean bimodal separation means the 0.5 threshold is not
delicate — you can slide it toward 0.4 to catch the borderline misses (trading a few more
false alarms) or toward 0.7 to cut false alarms (barely touching recall), and the sweep
in [`MODEL_REPORT.md`](MODEL_REPORT.md) quantifies each choice.

*Reproduce:* run `eval_resnet.py` to get per-patch scores, then plot `P(defective)`
histograms split by label. Figure: `resnet/confidence_distribution.png`.
