#!/usr/bin/env python3
"""
Mine a defect-TYPE patch dataset for the multi-class ResNet (Goal 2).

Unlike mine_patches.py (which makes good/bad tiles), this crops a patch around EACH
annotated defect and labels it by the defect's class, so the classifier learns to name
the defect type instead of just good/defective.

  * one patch per defect box, cropped (centered, then lightly shift/rotation jittered),
  * label = the YOLO class of that box,
  * split BY TEMPLATE (held-out board layouts, like mine_patches.py --heal).

Output (one folder per class, the layout data_multiclass.py expects):
  datasets/pcb_defect_types/{train,val,test}/<class_name>/*.jpg

Usage:
  python resnet/mine_defect_types.py                       # HRIPCB, 1024 crop -> 384 jpg
  python resnet/mine_defect_types.py --save-size 512       # higher-res patches (Goal 3 combo)
"""
import argparse, random, shutil, sys
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mine_patches as mp   # reuse crop_variant / _pku_images / _TPL / SOURCE_PREFIX

CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
OUT = mp.DS / "pcb_defect_types"
SEED = 42


def parse_boxes(label_path, W, H):
    """YOLO label -> [(class_id, cx_px, cy_px, w_px, h_px)]."""
    out = []
    p = Path(label_path)
    if p.exists():
        for line in p.read_text().splitlines():
            q = line.split()
            if len(q) >= 5:
                cid = int(float(q[0])); cx, cy, bw, bh = (float(v) for v in q[1:5])
                out.append((cid, cx * W, cy * H, bw * W, bh * H))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(mp.SOURCE_PREFIX), default="hripcb")
    ap.add_argument("--patch", type=int, default=1024, help="crop size (px) around the defect")
    ap.add_argument("--aug", type=int, default=4, help="jitter variants per defect")
    ap.add_argument("--shift-px", type=float, default=12.0)
    ap.add_argument("--rot-deg", type=float, default=12.0)
    ap.add_argument("--save-size", type=int, default=384, help="stored patch size (downscaled)")
    ap.add_argument("--save-fmt", choices=["jpg", "png"], default="jpg")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng = random.Random(SEED)
    prefix = mp.SOURCE_PREFIX[args.source]

    # collect boards + group by template, then split templates 6/2/2 (held-out layouts)
    recs = []
    for img, lab in mp._pku_images(prefix):
        m = mp._TPL.match(img.name)
        recs.append((img, lab, f"tpl_{m.group(1) if m else img.stem}"))
    groups = sorted({r[2] for r in recs})
    random.Random(SEED).shuffle(groups)
    n = len(groups); n_va = max(1, round(n * 0.2)); n_te = max(1, round(n * 0.2)); n_tr = n - n_va - n_te
    split_of = {g: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test")
                for i, g in enumerate(groups)}
    print("template split:", {g: split_of[g][:2] for g in groups})

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    counts = {sp: {c: 0 for c in CLASSES} for sp in ("train", "val", "test")}
    for img, lab, tid in recs:
        sp = split_of[tid]
        im = cv2.imread(str(img), cv2.IMREAD_COLOR)
        if im is None:
            continue
        H, W = im.shape[:2]
        for cid, dcx, dcy, bw, bh in parse_boxes(lab, W, H):
            if not (0 <= cid < len(CLASSES)):
                continue
            cls = CLASSES[cid]
            variants = [(dcx, dcy, 0.0)]
            for _ in range(args.aug):
                variants.append((dcx + rng.uniform(-args.shift_px, args.shift_px),
                                 dcy + rng.uniform(-args.shift_px, args.shift_px),
                                 rng.uniform(-args.rot_deg, args.rot_deg)))
            for cx, cy, ang in variants:
                patch = mp.crop_variant(im, cx, cy, args.patch, ang)
                if args.save_size and args.save_size != patch.shape[0]:
                    patch = cv2.resize(patch, (args.save_size, args.save_size),
                                       interpolation=cv2.INTER_AREA)
                d = out / sp / cls
                d.mkdir(parents=True, exist_ok=True)
                fn = f"{tid}_{cls}_{counts[sp][cls]:06d}.{args.save_fmt}"
                params = [cv2.IMWRITE_JPEG_QUALITY, 95] if args.save_fmt == "jpg" else []
                cv2.imwrite(str(d / fn), patch, params)
                counts[sp][cls] += 1

    print(f"\n[pcb_defect_types] -> {out}  (patch={args.patch} -> {args.save_size})")
    for sp in ("train", "val", "test"):
        print(f"  {sp:5s}: {counts[sp]}  total={sum(counts[sp].values())}")
    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n")
    print("Train with:  python resnet/train_multiclass.py --data datasets/pcb_defect_types "
          f"--size {min(args.save_size, 256)}")


if __name__ == "__main__":
    main()
