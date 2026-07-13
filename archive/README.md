# Archive — superseded experiments

The project settled on the **512×512 offset-trained ResNet-50** classifier
(`release_resnet_512/`). Everything that led there is parked here as zips —
the live working directories were deleted to reclaim ~57 GB.

Each zip restores to its original path (`unzip <file>` from the repo root).
The code and reports for all of this are still in the repo and in git history;
only the large binaries (datasets, weights, run outputs) were archived.

| archive | size | what it was |
|---|---|---|
| `datasets_pcb_bin_center_v2.zip` | 1.8G | Binary classifier, defects CENTERED — run / dataset (the ablation baseline) |
| `datasets_pcb_bin_offset_v2.zip` | 1.8G | Binary classifier, defects OFFSET — superseded run / dataset (256 + centered variants) |
| `datasets_pcb_defect_types7_indist.zip` | 1.3G | 7-way defect-type classifier — run / dataset |
| `datasets_pcb_defect_types7.zip` | 1.3G | 7-way defect-type classifier — run / dataset |
| `datasets_pcb_defect_types_indist.zip` | 678M | 7-way defect-type classifier — run / dataset |
| `datasets_pcb_defect_types.zip` | 678M | 7-way defect-type classifier — run / dataset |
| `datasets_pcb_patches_512_indist.zip` | 1.8G | 512px patch experiment (earlier generation) — run / dataset |
| `datasets_pcb_patches_512.zip` | 1.8G | 512px patch experiment (earlier generation) — run / dataset |
| `datasets_pcb_patches_offcenter_indist.zip` | 1.8G | Off-center position experiment — run / dataset |
| `datasets_pcb_patches_offcenter.zip` | 1.8G | Off-center position experiment — run / dataset |
| `datasets_pcb_types7_offset_v2.zip` | 1.3G | 7-way defect-type classifier — run / dataset |
| `pcb_patches_defectsplit.zip` | 1.1G | Original mined patch dataset / run (pre-split-fix) |
| `pcb_patches.zip` | 1.1G | Original mined patch dataset / run (pre-split-fix) |
| `pcb_sources.zip` | 1.1G | Raw source datasets (PKU / roboflow / DeepPCB) |
| `runs_resnet_at256_pcb_bin_center_v2.zip` | 666M | Binary classifier, defects CENTERED — run / dataset (the ablation baseline) |
| `runs_resnet_at256_pcb_bin_offset_v2.zip` | 665M | Binary classifier, defects OFFSET — superseded run / dataset (256 + centered variants) |
| `runs_resnet_at256_pcb_patches_512_indist.zip` | 666M | 512px patch experiment (earlier generation) — run / dataset |
| `runs_resnet_at256_pcb_patches_512.zip` | 665M | 512px patch experiment (earlier generation) — run / dataset |
| `runs_resnet_pcb_bin_center_v2.zip` | 667M | Binary classifier, defects CENTERED — run / dataset (the ablation baseline) |
| `runs_resnet_pcb_bin_offset_v2.zip` | 668M | Binary classifier, defects OFFSET — superseded run / dataset (256 + centered variants) |
| `runs_resnet_pcb_defect_types7_indist.zip` | 417M | 7-way defect-type classifier — run / dataset |
| `runs_resnet_pcb_defect_types7.zip` | 417M | 7-way defect-type classifier — run / dataset |
| `runs_resnet_pcb_defect_types_indist.zip` | 416M | 7-way defect-type classifier — run / dataset |
| `runs_resnet_pcb_defect_types.zip` | 415M | 7-way defect-type classifier — run / dataset |
| `runs_resnet_pcb_patches_512_indist.zip` | 667M | 512px patch experiment (earlier generation) — run / dataset |
| `runs_resnet_pcb_patches_512.zip` | 667M | 512px patch experiment (earlier generation) — run / dataset |
| `runs_resnet_pcb_patches_offcenter_indist.zip` | 665M | Off-center position experiment — run / dataset |
| `runs_resnet_pcb_patches_offcenter.zip` | 665M | Off-center position experiment — run / dataset |
| `runs_resnet_pcb_types7_offset_v2.zip` | 417M | 7-way defect-type classifier — run / dataset |
| `runs_resnet_v3_pcb_bin_center_256.zip` | 666M | Binary classifier, defects CENTERED — run / dataset (the ablation baseline) |
| `runs_resnet_v3_pcb_bin_center_512.zip` | 668M | Binary classifier, defects CENTERED — run / dataset (the ablation baseline) |
| `runs_resnet_v3_pcb_bin_offset_256.zip` | 665M | Binary classifier, defects OFFSET — superseded run / dataset (256 + centered variants) |
| `runs_resnet_v3_pcb_bin_offset_512.zip` | 668M | Binary classifier, defects OFFSET — superseded run / dataset (256 + centered variants) |
| `runs_yolov3_unified_gray640.zip` | 1.2G | YOLOv3 detector — trained run / 640px grayscale dataset (mAP@0.5 0.599) |
| `unified_pku_yolo_gray640.zip` | 2.8G | YOLOv3 detector — trained run / 640px grayscale dataset (mAP@0.5 0.599) |

## Still live (not archived)

| path | why |
|---|---|
| `runs_resnet_v3/pcb_bin_offset_512/` | the deployed model's weights |
| `datasets/pcb_bin_offset/` | its train/val/test split |
| `datasets/kaggle-hripcb/`, `datasets/clean_plates/` | raw source + healed plates, needed to re-mine patches |
| `release_resnet_512/` + `release_resnet512.zip` | the deployment bundle (IR + weights + test set) |
