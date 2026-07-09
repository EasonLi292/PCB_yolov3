#!/usr/bin/env python3
"""
Mine a defect-TYPE patch dataset for the multi-class ResNet (Goal 2).

Unlike mine_patches.py (which makes good/bad tiles), this crops a patch around EACH
annotated defect and labels it by the defect's class, so the classifier learns to name
the defect type instead of just good/defective.

  * one patch per defect box, labeled by the defect's YOLO class,
  * with --defect-offset F the defect is placed OFF-center by up to F*patch (uniform in
    both axes) so it can land ANYWHERE in the tile -- simulates a real board where a
    sliding window catches a defect at a random position (not conveniently centered),
  * with --good-per-plate N it ALSO emits a "good" (no-defect) class from each template's
    reconstructed clean plate (median of aligned copies, same as mine_patches.py --heal),
    turning this into a single {good + 6 defect types} classifier,
  * split BY TEMPLATE (held-out board layouts, like mine_patches.py --heal).

Output (one folder per class, the layout data_multiclass.py expects):
  datasets/pcb_defect_types/{train,val,test}/<class_name>/*.jpg

Usage:
  python resnet/mine_defect_types.py                                    # 6 defect classes, centered
  python resnet/mine_defect_types.py --defect-offset 0.4 --good-per-plate 150 \
         --out datasets/pcb_defect_types7                              # 7-class, defects anywhere
"""
import argparse, random, shutil, sys
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mine_patches as mp   # reuse crop_variant / good_tiles / build_clean_plates / _pku_images / _TPL

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
    ap.add_argument("--defect-offset", type=float, default=0.0,
                    help="place each defect off-center by up to this fraction of --patch "
                         "(0 = centered; ~0.4 simulates defects anywhere in the tile)")
    ap.add_argument("--good-per-plate", type=int, default=0,
                    help="also emit a GOOD (no-defect) class: this many clean tiles per "
                         "template clean-plate (0 = defects-only, the original behavior)")
    ap.add_argument("--plate-n", type=int, default=25, help="copies to median per clean plate")
    ap.add_argument("--min-std", type=float, default=12.0, help="min grayscale std for a GOOD tile")
    ap.add_argument("--split-mode", choices=["template", "defect"], default="template",
                    help="template = hold out whole board layouts; defect = split individual "
                         "defects/tiles so the SAME designs appear in train AND test (in-distribution)")
    ap.add_argument("--save-size", type=int, default=384, help="stored patch size (downscaled)")
    ap.add_argument("--save-fmt", choices=["jpg", "png"], default="jpg")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng = random.Random(SEED)
    prefix = mp.SOURCE_PREFIX[args.source]
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if args.save_fmt == "jpg" else []

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
    all_classes = (["good"] if args.good_per_plate > 0 else []) + CLASSES
    counts = {sp: {c: 0 for c in all_classes} for sp in ("train", "val", "test")}

    # ---- defect-TYPE patches: one per box, placed off-center by up to defect_offset*patch ----
    off = args.defect_offset * args.patch
    for img, lab, tid in recs:
        im = cv2.imread(str(img), cv2.IMREAD_COLOR)
        if im is None:
            continue
        H, W = im.shape[:2]
        for bi, (cid, dcx, dcy, bw, bh) in enumerate(parse_boxes(lab, W, H)):
            if not (0 <= cid < len(CLASSES)):
                continue
            cls = CLASSES[cid]
            sp = (split_of[tid] if args.split_mode == "template"
                  else mp.unit_split(f"{tid}:{Path(img).stem}", cls, bi))   # per (photo, defect)
            variants = []
            for k in range(args.aug + 1):
                if off:                                  # defect anywhere in the tile
                    ox, oy = rng.uniform(-off, off), rng.uniform(-off, off)
                    ang = rng.uniform(-args.rot_deg, args.rot_deg)
                elif k == 0:                             # original behaviour: centered base ...
                    ox = oy = ang = 0.0
                else:                                    # ... plus small shift/rotate jitters
                    ox, oy = (rng.uniform(-args.shift_px, args.shift_px),
                              rng.uniform(-args.shift_px, args.shift_px))
                    ang = rng.uniform(-args.rot_deg, args.rot_deg)
                variants.append((dcx + ox, dcy + oy, ang))
            for cx, cy, ang in variants:
                patch = mp.crop_variant(im, cx, cy, args.patch, ang)
                if args.save_size and args.save_size != patch.shape[0]:
                    patch = cv2.resize(patch, (args.save_size, args.save_size),
                                       interpolation=cv2.INTER_AREA)
                d = out / sp / cls
                d.mkdir(parents=True, exist_ok=True)
                fn = f"{tid}_{cls}_{counts[sp][cls]:06d}.{args.save_fmt}"
                cv2.imwrite(str(d / fn), patch, params)
                counts[sp][cls] += 1

    # ---- optional GOOD (no-defect) class: clean tiles from each template's healed plate ----
    if args.good_per_plate > 0:
        print("\nbuilding clean plates for the GOOD (no-defect) class ...")
        plates = mp.build_clean_plates(prefix, args.plate_n)
        for tid, ppath in plates.items():
            tpl = f"tpl_{tid}"
            sp0 = split_of.get(tpl)
            if sp0 is None:
                continue
            plate = cv2.imread(str(ppath), cv2.IMREAD_COLOR)
            if plate is None:
                continue
            for unit, patch in mp.good_tiles(plate, [], args, rng, args.good_per_plate):
                sp = (sp0 if args.split_mode == "template"
                      else mp.unit_split(f"{tpl}:{ppath.stem}", "good", unit))
                if args.save_size and args.save_size != patch.shape[0]:
                    patch = cv2.resize(patch, (args.save_size, args.save_size),
                                       interpolation=cv2.INTER_AREA)
                d = out / sp / "good"
                d.mkdir(parents=True, exist_ok=True)
                fn = f"{tpl}_good_{counts[sp]['good']:06d}.{args.save_fmt}"
                cv2.imwrite(str(d / fn), patch, params)
                counts[sp]["good"] += 1

    print(f"\n[pcb_defect_types] -> {out}  (patch={args.patch} -> {args.save_size}, "
          f"defect_offset={args.defect_offset}, classes={len(all_classes)})")
    for sp in ("train", "val", "test"):
        print(f"  {sp:5s}: {counts[sp]}  total={sum(counts[sp].values())}")
    (out / "classes.txt").write_text("\n".join(sorted(all_classes)) + "\n")

    import json, subprocess, datetime, sys as _sys
    info = {
        "kind": "pcb_defect_types", "classes": sorted(all_classes), "source": args.source,
        "split_mode": args.split_mode,
        "patch": args.patch, "save_size": args.save_size, "save_fmt": args.save_fmt,
        "aug": args.aug, "shift_px": args.shift_px, "rot_deg": args.rot_deg, "seed": SEED,
        "defect_offset": args.defect_offset, "good_per_plate": args.good_per_plate,
        "plate_n": args.plate_n, "min_std": args.min_std,
        "counts": counts, "template_split": {g: split_of[g] for g in sorted(split_of)},
        "argv": " ".join(_sys.argv),
        "regenerate": "re-run this exact command (mining is seeded/deterministic)",
    }
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(mp.ROOT)).decode().strip()
    except Exception:
        info["git_commit"] = "unknown"
    info["created"] = datetime.datetime.now().isoformat(timespec="seconds")
    (out / "dataset_manifest.json").write_text(json.dumps(info, indent=2))
    print(f"wrote manifest -> {out / 'dataset_manifest.json'}")
    print("Train with:  python resnet/train_multiclass.py --data " + str(out) +
          f" --size {min(args.save_size, 256)}")


if __name__ == "__main__":
    main()
