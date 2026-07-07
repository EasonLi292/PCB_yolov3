# Experiment Plan — 4 goals

Continuation work for the PCB inspection project, to run on the GPU box after unzipping
`pcb_sources.zip` and rebuilding the datasets (see [`resnet/README.md`](resnet/README.md)
→ *Regenerating the dataset from the source boards*). Each goal below lists the objective,
the concrete commands, the metrics to report, and the deliverable.

Prereqs on the new machine:
```bash
git clone https://github.com/EasonLi292/PCB_yolov3.git && cd PCB_yolov3
pip install -r requirements-train.txt
gdown <PCB_SOURCES_DRIVE_ID> -O datasets/pcb_sources.zip
cd datasets && unzip -q pcb_sources.zip && cd ..
python yolov3/build_unified_yolo.py        # -> datasets/unified_pku_yolo
python yolov3/preprocess_images.py         # -> datasets/unified_pku_yolo_gray640 (YOLO input)
```

---

## Goal 1 — Finish YOLO training & compare to the ResNet baseline

**Objective.** Take the detector past the 0.408 mAP@0.5 baseline (7-epoch, frozen
backbone), then compare it to the ResNet good/bad classifier **on the same terms** —
board-level good/bad, including the false-alarm rate on clean boards that mAP never shows.

**Why a special comparison.** YOLO outputs boxes (measured by mAP); ResNet outputs a
board verdict (measured by accuracy). The fair common ground is **board-level good/bad**:
a board is BAD if the detector emits ≥1 detection. `yolov3/board_level_eval.py` computes
exactly that and lines the numbers up against `resnet/MODEL_REPORT.md`.

**Steps.**
```bash
# 1a. Finish training: two-phase, unfreeze the Darknet-53 backbone, more epochs.
python yolov3/train_yolov3.py --data datasets/unified_pku_yolo_gray640 \
       --epochs 40 --batch 8                                   # phase 1 (frozen) if not done
python yolov3/train_yolov3.py --data datasets/unified_pku_yolo_gray640 \
       --resume runs/unified_pku_yolo_gray640/best.weights.h5 \
       --unfreeze --lr 1e-4 --epochs 60                        # phase 2 (fine-tune)
#     (optional but recommended) re-cluster anchors first:
python yolov3/compute_anchors.py --data datasets/unified_pku_yolo_gray640

# 1b. Standard detection metric (per-class AP + mAP@0.5):
python yolov3/export_fpga.py    --weights runs/.../best.weights.h5 --out runs/.../openvino_fpga
python yolov3/analyze_openvino.py --ir runs/.../openvino_fpga/yolov3_fpga_fp32.xml \
       --data datasets/unified_pku_yolo_gray640 --split test --classes release/classes.txt

# 1c. Board-level good/bad + false-alarm rate on GOOD boards (the comparison instrument):
python yolov3/board_level_eval.py \
       --ir runs/.../openvino_fpga/yolov3_fpga_fp32.xml \
       --defective datasets/unified_pku_yolo_gray640 --split test \
       --good "datasets/deeppcb/PCBData/**/*_temp.jpg" "datasets/clean_plates/plate_*.png" \
       --classes release/classes.txt --score 0.25
```

**Metrics to report.** mAP@0.5 + per-class AP (1b); board-level accuracy, recall
(defects caught), and **false-alarm rate on good boards** (1c). Put the board-level trio
next to the ResNet numbers (acc 0.981 / recall 0.997 patch-level → aggregate to board).

**Deliverable.** Updated `MODEL_REPORT.md` with the new mAP and a "detector vs classifier,
board-level" table. Sweep `--score` to trade recall vs false alarms.

---

## Goal 2 — ResNet that classifies the defect TYPE (not just good/bad)

**Objective.** A 6-class classifier (`missing_hole, mouse_bite, open_circuit, short, spur,
spurious_copper`) instead of binary. Same ResNet-50 backbone, softmax head.

**What's new (already in the repo).**
- `resnet/mine_defect_types.py` — crops one patch per annotated defect, labeled by class,
  split by template.
- `build_resnet50(n_classes=6)` — softmax head (added to `resnet/resnet50_tf.py`).
- `resnet/train_multiclass.py`, `resnet/eval_multiclass.py`, `resnet/data_multiclass.py`.

**Steps.**
```bash
# 2a. Mine the per-class defect patch dataset (HRIPCB, by-template split):
python resnet/mine_defect_types.py                         # -> datasets/pcb_defect_types

# 2b. Two-phase training (softmax over 6 classes):
python resnet/train_multiclass.py --data datasets/pcb_defect_types --size 256 --epochs 25
python resnet/train_multiclass.py --data datasets/pcb_defect_types --size 256 \
       --resume runs_resnet/pcb_defect_types/best.weights.h5 --unfreeze --lr 1e-5 --epochs 20

# 2c. Evaluate: top-1 accuracy, 6x6 confusion matrix, per-class P/R/F1:
python resnet/eval_multiclass.py --weights runs_resnet/pcb_defect_types/best.weights.h5 \
       --data datasets/pcb_defect_types --size 256
```

**Metrics.** Top-1 accuracy, macro-F1, the 6×6 confusion matrix (which types get confused —
expect mouse_bite↔open_circuit and spur↔spurious_copper overlap), per-class recall.

**Deliverable.** `resnet/MODEL_REPORT_MULTICLASS.md` with the confusion matrix + per-class
table. **Note:** this dataset is defects-only; keep the binary good/bad model for the
"is there a defect at all" decision, and use this one to name the defect once found (a
natural two-stage detector→type or good/bad→type pipeline).

---

## Goal 3 — ResNet at 512×512 and how accuracy changes

**Objective.** Quantify the effect of resolution. Current patches are a 1024 px crop
squished to 384 and trained at 256; at 512 the crop is only 2× downscaled — more defect
signal for the tiny defects that dominate the misses.

**Steps.**
```bash
# 3a. Mine the SAME good/bad patches but store at 512 (into a separate dir):
python resnet/mine_patches.py --heal --save-size 512 --out datasets/pcb_patches_512

# 3b. Train + eval at size 512 (two-phase, as usual):
python resnet/train_resnet.py --data datasets/pcb_patches_512 --size 512 --epochs 20 --batch 16
python resnet/train_resnet.py --data datasets/pcb_patches_512 --size 512 --batch 16 \
       --resume runs_resnet/pcb_patches_512/best.weights.h5 --unfreeze --lr 1e-5 --epochs 15
python resnet/eval_resnet.py --weights runs_resnet/pcb_patches_512/best.weights.h5 \
       --data datasets/pcb_patches_512 --size 512
```

**Metrics.** Compare @512 vs the @256 baseline on the **same held-out split**: accuracy,
recall, precision, PR-AUC, and especially the count/confidence of the confident misses
(re-run `resnet/*` confidence analysis). Also record inference cost (bigger input = more
DLA cost) — 512² is 4× the compute of 256².

**Deliverable.** A short `resnet/RESOLUTION_REPORT.md`: 256 vs 512 metrics side-by-side +
the accuracy-vs-cost trade-off, and a recommendation (256/384/512) for the FPGA target.
Remember to re-export the IR at `--size 512` if you deploy it.

---

## Goal 4 — Vary defect position; does distance-to-center affect accuracy?

**Objective.** We found the current model is **not** position-invariant (confidence halves
when a defect moves ~20% off-center) because every training defect was centered. Train with
off-center defects and re-measure the position curve — it should flatten.

**What's new (already in the repo).**
- `resnet/mine_patches.py --defect-offset FRAC` — places each defect off-center by up to
  `FRAC × patch` during mining (position augmentation).
- `resnet/offcenter_test.py` — measures P(defective) vs the defect's offset from center
  (crops straight from the labelled source, so the defect location is exact).

**Steps.**
```bash
# 4a. Baseline curve on the CURRENT (center-trained) model:
python resnet/offcenter_test.py --weights best.weights.h5 --size 256 --source hr_08 \
       --out resnet/offcenter_before.png

# 4b. Re-mine with off-center defects, retrain:
python resnet/mine_patches.py --heal --defect-offset 0.3 --out datasets/pcb_patches_offcenter
python resnet/train_resnet.py --data datasets/pcb_patches_offcenter --size 256 --epochs 20
python resnet/train_resnet.py --data datasets/pcb_patches_offcenter --size 256 \
       --resume runs_resnet/pcb_patches_offcenter/best.weights.h5 --unfreeze --lr 1e-5 --epochs 15

# 4c. Curve on the position-augmented model:
python resnet/offcenter_test.py \
       --weights runs_resnet/pcb_patches_offcenter/best.weights.h5 --size 256 --source hr_08 \
       --out resnet/offcenter_after.png
```

**Metrics.** The P(defective)-vs-offset table + plot, before vs after. Success = the
"after" curve stays high (≥ ~0.9) out to ~30–40% offset instead of collapsing at 20%.
Also re-check overall test accuracy didn't regress.

**Deliverable.** `resnet/POSITION_REPORT.md` with both curves and the verdict on whether
position augmentation removes the center bias (it validates the sliding-window deployment
assumption — a defect landing anywhere in a tile still gets caught).

---

## Saved artifacts & reproducibility (re-test later without re-training)

Every training and mining run now writes a **manifest** so the weights, config, and the
exact dataset are recorded — you can come back months later and evaluate without retraining.

- **Training** (`train_resnet.py`, `train_multiclass.py`, `train_yolov3.py`) saves, under
  `runs_resnet/<dataset>/` (or `runs/<dataset>/` for YOLO):
  `best.weights.h5` (guaranteed written, restored-best), `saved_model/`, `classes.txt`, and
  **`run_manifest.json`** — task, dataset path, size/batch/epochs/lr, phase, augment flag,
  class counts, best val metric, git commit, timestamp, and the exact re-test command.
- **Mining** (`mine_patches.py`, `mine_defect_types.py`) writes **`dataset_manifest.json`**
  into the dataset dir — source, all mining params, seed, per-split counts, template split,
  and the argv. Mining is **seeded/deterministic**, so the same command reproduces the same
  dataset byte-for-byte.

**To preserve a dataset for future testing, keep either:**
1. the `dataset_manifest.json` (then re-run its `argv` on the same source boards to
   regenerate it exactly — smallest to keep), **or**
2. a zip of the dataset dir on Drive, if you want the test split without re-mining:
   ```bash
   zip -qr datasets/pcb_patches_512.zip datasets/pcb_patches_512   # then upload to Drive
   ```

**Keep** `runs_*/` (weights + manifests) — they're small and are all you need to re-run
`eval_*.py` / `board_level_eval.py` later. `runs_resnet/` is gitignored (big weights stay
out of git); back it up to Drive alongside the dataset zips.

## Suggested order & notes
- Independent, but a sensible order is **3 → 4 → 2 → 1** (3 & 4 reuse the binary pipeline;
  2 adds a head; 1 is the largest / detector-side effort).
- All ResNet scripts share the two-phase freeze→unfreeze recipe and the `--size` knob.
- Re-export the FPGA IR whenever the input size or task changes
  (`resnet/export_openvino.py --size <N>`; detector via `yolov3/export_fpga.py`).
- New/added code for these goals: `resnet/{mine_defect_types,train_multiclass,eval_multiclass,
  data_multiclass,offcenter_test}.py`, `build_resnet50(n_classes=...)`,
  `mine_patches.py --defect-offset`, and `yolov3/board_level_eval.py`.
