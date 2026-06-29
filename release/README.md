# release/ — deployable model artifacts

This is the easy-to-find home for the **trained, FPGA-ready model**. After an H100
training run + FPGA export, the deployable files live here:

| File | What it is |
|------|------------|
| `yolov3_fpga_fp32.xml` / `.bin` | OpenVINO IR (raw conv heads) — **feed this to the Intel FPGA AI Suite compiler** |
| `classes.txt` | the 6 class names, in index order |
| `anchors.json` | the k-means anchors (also needed by host-side decode) |

Host-side decode + NMS: `scripts/yolo_postprocess.py` (numpy, no TF).

The `.xml`/`.bin`/`.h5` are **not committed** (too large for git) — only this README is
tracked. Populate this folder with:

```bash
# after export, copy the IR here:
cp runs/unified_pku_yolo_gray640/openvino_fpga/yolov3_fpga_fp32.* release/
cp runs/unified_pku_yolo_gray640/openvino_fpga/classes.txt release/
cp scripts/anchors.json release/
```

Then refresh the report with the new model:

```bash
python scripts/make_report.py \
    --ir release/yolov3_fpga_fp32.xml --data datasets/unified_pku_yolo_gray640 \
    --classes release/classes.txt --trained "30 epochs frozen + 20 unfrozen (H100)"
```
