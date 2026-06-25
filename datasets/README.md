# PCB Defect Datasets

Six PCB-defect sources loaded and normalized into **three ready-to-train YOLO datasets
that all share one identical layout**, so they load and split the same way.
Built by [`scripts/build_unified_yolo.py`](../scripts/build_unified_yolo.py).

## Consistent layout (every trainable dataset)

```
datasets/<name>/
    data.yaml
    train/images/   train/labels/
    val/images/     val/labels/
    test/images/    test/labels/
```

Point any YOLO trainer (Ultralytics, YOLOv5/v8, etc.) at `datasets/<name>/data.yaml`.

| Dataset | `data.yaml` | Images (train/val/test) | Classes |
|---------|-------------|-------------------------|---------|
| **`unified_pku_yolo/`** ⭐ | `unified_pku_yolo/data.yaml` | 14,664 (12,084 / 1,294 / 1,286) | 6 |
| **`deeppcb_yolo/`** | `deeppcb_yolo/data.yaml` | 1,500 (1,200 / 150 / 150) | 6 |

- **6-class taxonomy** (both): `0 missing_hole · 1 mouse_bite · 2 open_circuit · 3 short · 4 spur · 5 spurious_copper`

**`unified_pku_yolo` now includes DeepPCB in TRAIN only** (1,500 `dp_*` images): the 3 PKU
sources (norbertelter + HRIPCB + Roboflow) are group-split into train/val/test, then the
DeepPCB defect images are appended to **train** as extra data. DeepPCB is a different visual
domain (binary 1-bit boards), so it's kept out of val/test — the **val/test splits are
byte-identical to before**, keeping the benchmark comparable. Use `deeppcb_yolo` to train a
DeepPCB-only model or as a cross-domain test set.

**DsPCBSD+ was discarded** — its 9-class taxonomy can't merge cleanly (no `missing_hole`,
adds hole_breakout/scratch/foreign-object classes); a partial merge would leave those
defects unlabeled and teach false negatives.

Training applies **online augmentation** (flips, 90° rotation, brightness/contrast — see
`augment()` in [`scripts/yolov3_tf.py`](../scripts/yolov3_tf.py)); disable with `--no-augment`.

## Splitting — consistent, deterministic, leakage-free

Splits are produced by the build script with a single config (top of the file):

```python
SPLIT = (0.80, 0.10, 0.10)   # train / val / test
SEED  = 42
```

- **Deterministic:** same seed → same split every run. Edit `SPLIT`/`SEED` and re-run to
  re-split all datasets identically.
- **Group-aware (PKU):** the 13,164 PKU images are augmentations of only **693 original
  boards**. The splitter groups by `<board>_<defect>_<instance>` (shared across every
  augmentation *and* across the three source datasets) so no original board appears in
  more than one split — **verified 0 groups span splits**. This removes the train/val
  leakage that a naive random split would cause.

Re-split everything: `python scripts/build_unified_yolo.py all`
(or `pku` / `deeppcb` / `dspcbsd` individually).

## Preprocessed (network-ready) variants

Run by [`scripts/preprocess_images.py`](../scripts/preprocess_images.py). Each trainable
dataset has a `*_gray640/` twin where every image is **grayscale, single-channel, and
letterboxed to 640×640** (aspect ratio preserved, padded with 114), with labels
transformed to match. Same `{train,val,test}/{images,labels}` + `data.yaml` layout.

| Network-ready dataset | Images | Format |
|-----------------------|--------|--------|
| `unified_pku_yolo_gray640/` | 13,164 | 1-channel PNG, 640×640 |
| `deeppcb_yolo_gray640/` | 1,500 | 1-channel PNG, 640×640 |
| `dspcbsd_yolo_gray640/` | 10,259 | 1-channel PNG, 640×640 |

All verified: every image 640×640×1, all boxes in range, box counts identical to the
source sets (no labels lost). Train with single-channel input (e.g. Ultralytics `ch=1`).

```bash
python scripts/preprocess_images.py all            # build all *_gray640 sets
python scripts/preprocess_images.py pku --size 512 # custom size / single dataset
```

Note: if your trainer already letterboxes on the fly (Ultralytics does), you can train
straight off the color `*_yolo/` sets with `imgsz=640`; the `*_gray640/` sets are for a
fixed grayscale single-channel pipeline as requested.

## Where the data came from (provenance)

The three builds are derived from these raw sources, kept untouched alongside:

| Build | Raw source folder | Origin | License / notes |
|-------|-------------------|--------|-----------------|
| `unified_pku_yolo` | `kaggle-pcb-defect/` | Kaggle — https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset | Augmented PKU set, 10,668 imgs. Byte-identical to `voc-pcb-augmented`. |
| `unified_pku_yolo` | `kaggle-hripcb/` | Kaggle — https://www.kaggle.com/datasets/youssefhassan12/hripcb-dataset | Original high-res PKU/HRIPCB, 693 imgs (the parent boards). |
| `unified_pku_yolo` | `roboflow-pcb/` | Roboflow Universe — https://universe.roboflow.com/pcbdataset/pcb-defect-detection-9ewqw (v2) | 1,803 imgs. CC BY 4.0. |
| `deeppcb_yolo` | `deeppcb/` | GitHub — https://github.com/Charmve/Surface-Defect-Detection/tree/master/DeepPCB | 1,500 grayscale 640×640 template/test pairs; custom `x1 y1 x2 y2 type` annotations. |
| `dspcbsd_yolo` | `kaggle-dspcbsd/` | Kaggle — https://www.kaggle.com/datasets/enisteper1/dataset-of-pcb-surface-defects-dspcbsd | DsPCBSD+, 10,259 imgs, 9-class surface taxonomy. |

Also on disk (not part of a build):

| Folder | Origin | Notes |
|--------|--------|-------|
| `voc-pcb-augmented/` | Dropbox `VOC_PCB.zip`, linked from the TDD-Net README | VOC XML version of the same data as `kaggle-pcb-defect`. Kept for the XML annotations. Contains a junk `idaneel` box class (non-defect) that is dropped on conversion. |
| `tiny-defect-detection/` | GitHub — https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB | TDD-Net (FPN) **training code**, not image data. Underlying dataset is the PKU set above (Peking University Open Lab on Human-Robot Interaction). |

**Lineage note:** the PKU sources (norbertelter, HRIPCB, Roboflow, VOC_PCB) all trace
back to the **Peking University HRIPCB** dataset — 693 original boards with 6 photoshopped
defect types — which is why grouping by board id deduplicates them cleanly. DeepPCB and
DsPCBSD+ are independent datasets.

## Class-index remapping (why you can't just `cat` the raw labels)

The raw PKU sources use **different index orders for the same classes** — e.g. index `0`
is `mouse_bite` in norbertelter but `missing_hole` in HRIPCB. The build script remaps
every source by class *name* into the canonical order above. Always rebuild via the
script rather than merging raw label folders by hand.

## DsPCBSD+ 9 classes
`SH` short · `SP` spur · `SC` spurious_copper · `OP` open · `MB` mouse_bite ·
`HB` hole_breakout · `CS` conductor_scratch · `CFO` conductor_foreign_object ·
`BMFO` base_material_foreign_object
