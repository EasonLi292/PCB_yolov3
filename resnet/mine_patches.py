#!/usr/bin/env python3
"""
Mine a PATCH-level good/bad dataset from boards we already have.

Idea: a defective board is ~99% defect-free *area*. Slide a zoomed-in window over
each board:
  * a tile that does NOT touch any defect box  -> a clean GOOD patch
  * a tile centered on a defect box            -> a BAD patch
Good and bad therefore come from the SAME boards at the SAME zoom -> no domain leak
(the model can't cheat on dataset/scale, only on the defect itself). Each kept tile is
augmented with slight variations (sub-pixel shift + small rotation).

--- HEAL MODE (recommended for HRIPCB) ---
PKU/HRIPCB defects are photoshopped onto 10 base templates; each template has many
PIXEL-ALIGNED copies, each with a small defect in a *different* place. The per-pixel
MEDIAN across a template's copies washes the defects out and reconstructs a fully
CLEAN board ("clean plate"). With --heal:
  * GOOD patches come from sliding over the whole CLEAN plate (every location valid,
    guaranteed defect-free -- the entire board becomes good data).
  * BAD patches come from the defective images, cropped around each annotated defect.
  * Split is by TEMPLATE, so val/test are held-out board layouts (honest, no leak).

Sources (boards + defect boxes):
  * PKU (default) -- datasets/unified_pku_yolo COLOR boards (norbertelter + HRIPCB +
    Roboflow) with YOLO labels.  --source picks one; default HRIPCB (high-res).
  * DeepPCB (--include-deeppcb) -- the binary BLACK-AND-WHITE registered set.

Output (color, same layout the classifier expects):
  datasets/pcb_patches/{train,val,test}/{good,bad}/*.png
Clean plates are cached under datasets/clean_plates/.

Usage:
  python resnet/mine_patches.py --heal                       # HRIPCB healed plates, 1024
  python resnet/mine_patches.py --heal --build-plates-only   # just build/inspect plates
  python resnet/mine_patches.py --source all --patch 512     # non-heal, all color sources
"""
import argparse, random, re, shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "datasets"
OUT = DS / "pcb_patches"
PLATES = DS / "clean_plates"

SPLIT = (0.80, 0.10, 0.10)   # train / val / test  (matches the rest of the repo)
SEED = 42
_PKU_CORE = re.compile(r"(\d+_[a-z_]+_\d+)", re.I)   # board core shared across augmentations
_TPL = re.compile(r"hr_(\d+)_", re.I)                # HRIPCB template id
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SOURCE_PREFIX = {"hripcb": "hr_", "norbertelter": "nb_", "roboflow": "rf_", "all": None}


# ----------------------------- board collection -----------------------------
# A board record = (img_path, boxes_or_marker, group_key)
def _pku_images(prefix=None):
    src = DS / "unified_pku_yolo"
    if not src.is_dir():
        return
    for sp in ("train", "val", "test"):
        idir, ldir = src / sp / "images", src / sp / "labels"
        if not idir.is_dir():
            continue
        for img in sorted(idir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS or img.name.startswith("dp_"):
                continue                                   # skip the B&W DeepPCB merge
            if prefix and not img.name.startswith(prefix):
                continue
            yield img, ldir / f"{img.stem}.txt"


def collect_pku(prefix=None):
    out = []
    for img, lab in _pku_images(prefix):
        m = _PKU_CORE.search(img.stem)
        grp = "pku_" + (m.group(1).lower() if m else img.stem)
        out.append((img, "yolo:" + str(lab), grp))
    return out


def collect_deeppcb():
    base = DS / "deeppcb" / "PCBData"
    out = []
    if not base.is_dir():
        return out
    for p in base.rglob("*_test.jpg"):
        sub = p.parent.name
        idd = p.stem[:-5]
        ann = p.parent.parent / f"{sub}_not" / f"{idd}.txt"
        boxes = []
        if ann.exists():
            for line in ann.read_text().splitlines():
                q = line.split()
                if len(q) >= 4:
                    x1, y1, x2, y2 = map(float, q[:4])
                    boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
        out.append((p, boxes, f"dp_{idd}"))
    for p in base.rglob("*_temp.jpg"):
        out.append((p, [], f"dp_{p.stem[:-5]}"))
    return out


def _pku_boxes(label_marker, W, H):
    path = Path(label_marker.split("yolo:", 1)[1])
    boxes = []
    if path.exists():
        for line in path.read_text().splitlines():
            q = line.split()
            if len(q) >= 5:
                cx, cy, bw, bh = (float(v) for v in q[1:5])
                boxes.append(((cx - bw / 2) * W, (cy - bh / 2) * H,
                              (cx + bw / 2) * W, (cy + bh / 2) * H))
    return boxes


# ----------------------------- clean plates (heal) -----------------------------
def build_clean_plates(prefix, n, force=False):
    """Median across each HRIPCB template's aligned copies -> clean plate per template.
    Returns {template_id: plate_path}. Cached under datasets/clean_plates/."""
    PLATES.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for img, _ in _pku_images(prefix):
        m = _TPL.match(img.name)
        if m:
            groups[m.group(1)].append(img)
    plates = {}
    for tid in sorted(groups):
        outp = PLATES / f"plate_{tid}.png"
        if outp.exists() and not force:
            plates[tid] = outp
            print(f"  template {tid}: cached plate"); continue
        # stack up to n copies sharing the majority dimension
        stack, shp = [], None
        for p in groups[tid]:
            im = cv2.imread(str(p))
            if im is None:
                continue
            if shp is None:
                shp = im.shape
            if im.shape != shp:
                continue
            stack.append(im)
            if len(stack) >= n:
                break
        arr = np.stack(stack)                              # (k,H,W,3) uint8
        med = np.empty(shp, np.uint8)
        for y in range(0, shp[0], 256):                    # strip median to bound memory
            med[y:y + 256] = np.median(arr[:, y:y + 256], axis=0).astype(np.uint8)
        cv2.imwrite(str(outp), med)
        plates[tid] = outp
        print(f"  template {tid}: plate from {len(stack)} copies {shp[1]}x{shp[0]}")
    return plates


# ----------------------------- patch extraction -----------------------------
def crop_variant(img, cx, cy, size, angle):
    """Zoomed-in size×size COLOR crop centered at (cx,cy), rotated by `angle`."""
    big = int(np.ceil(size * 1.5))
    patch = cv2.getRectSubPix(img, (big, big), (float(cx), float(cy)))
    if angle:
        M = cv2.getRotationMatrix2D((big / 2.0, big / 2.0), angle, 1.0)
        patch = cv2.warpAffine(patch, M, (big, big),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    s = (big - size) // 2
    return patch[s:s + size, s:s + size]


def _emit(img, cx, cy, args, rng, n_aug):
    yield crop_variant(img, cx, cy, args.patch, 0.0)        # base
    for _ in range(n_aug):
        dx = rng.uniform(-args.shift_px, args.shift_px)
        dy = rng.uniform(-args.shift_px, args.shift_px)
        ang = rng.uniform(-args.rot_deg, args.rot_deg)
        yield crop_variant(img, cx + dx, cy + dy, args.patch, ang)


def _overlaps(win, boxes, margin):
    x0, y0, x1, y1 = win
    for bx0, by0, bx1, by1 in boxes:
        if not (x1 + margin <= bx0 or bx1 <= x0 - margin or
                y1 + margin <= by0 or by1 <= y0 - margin):
            return True
    return False


def load_board(rec, args):
    path, boxes, _ = rec
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    H, W = img.shape[:2]
    if isinstance(boxes, str):
        boxes = _pku_boxes(boxes, W, H)
    ps = args.patch
    if H < ps or W < ps:                                    # upscale tiny boards; scale boxes
        sx, sy = max(W, ps) / W, max(H, ps) / H
        img = cv2.resize(img, (max(W, ps), max(H, ps)))
        boxes = [(b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy) for b in boxes]
    return img, boxes


def good_tiles(img, boxes, args, rng, cap):
    """Clean tiles sampled by RANDOM translation across the board ('translate across'):
    keep windows clear of any defect (boxes) and not near-uniform (so the model can't
    cheat on 'structure present => bad'). Continuous centers give far more distinct good
    crops than a coarse grid -- important when a big patch only fits ~2-3 times per board."""
    H, W = img.shape[:2]
    ps = args.patch
    clear = int(0.3 * ps) + args.shift_px
    lox, hix = ps / 2.0, max(ps / 2.0, W - ps / 2.0)
    loy, hiy = ps / 2.0, max(ps / 2.0, H - ps / 2.0)
    kept, tries = 0, 0
    while kept < cap and tries < cap * 25:
        tries += 1
        cx, cy = rng.uniform(lox, hix), rng.uniform(loy, hiy)
        x0, y0 = int(cx - ps / 2), int(cy - ps / 2)
        if _overlaps((x0, y0, x0 + ps, y0 + ps), boxes, clear):
            continue
        tile = img[max(0, y0):y0 + ps, max(0, x0):x0 + ps]
        if tile.size == 0 or float(cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY).std()) < args.min_std:
            continue
        yield from _emit(img, cx, cy, args, rng, args.aug)
        kept += 1


def bad_tiles(img, boxes, args, rng):
    """One window per defect, jittered a few times. With --defect-offset F, the window is
    shifted so the defect lands OFF-center (up to F*patch away) instead of centered — this
    is the position augmentation for Goal 4 (teach the model defects can be anywhere)."""
    off = getattr(args, "defect_offset", 0.0) * args.patch
    for bx0, by0, bx1, by1 in boxes:
        cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
        if off:
            cx += rng.uniform(-off, off)
            cy += rng.uniform(-off, off)
        yield from _emit(img, cx, cy, args, rng, args.bad_jitter)


# ----------------------------- driver -----------------------------
def _materialize(out, records, split_of, cap, args, rng):
    """records: list of (kind, board_rec). kind in {'good','bad'}."""
    if out.exists():
        shutil.rmtree(out)
    counts = {sp: {"good": 0, "bad": 0} for sp in ("train", "val", "test")}
    random.Random(SEED + 1).shuffle(records)
    for kind, rec in records:
        sp = split_of[rec[2]]
        if counts[sp][kind] >= cap[sp][kind]:
            continue
        img, boxes = load_board(rec, args)
        if img is None:
            continue
        gen = good_tiles(img, [] if kind == "good" else boxes, args, rng,
                         args.good_per_plate if kind == "good" else 0) if kind == "good" \
            else bad_tiles(img, boxes, args, rng)
        for patch in gen:
            if counts[sp][kind] >= cap[sp][kind]:
                break
            if args.save_size and args.save_size != patch.shape[0]:
                patch = cv2.resize(patch, (args.save_size, args.save_size),
                                   interpolation=cv2.INTER_AREA)
            d = out / sp / kind
            d.mkdir(parents=True, exist_ok=True)
            fn = f"{rec[2]}_{counts[sp][kind]:06d}.{args.save_fmt}"
            params = [cv2.IMWRITE_JPEG_QUALITY, 95] if args.save_fmt == "jpg" else []
            cv2.imwrite(str(d / fn), patch, params)
            counts[sp][kind] += 1
    return counts


def _report(counts, patch):
    print(f"[pcb_patches] -> {OUT}  (patch={patch}, color)")
    for sp in ("train", "val", "test"):
        g, b = counts[sp]["good"], counts[sp]["bad"]
        print(f"  {sp:5s}: good={g:6d}  bad={b:6d}")
    tg = sum(counts[sp]["good"] for sp in counts)
    tb = sum(counts[sp]["bad"] for sp in counts)
    print(f"  TOTAL: good={tg}  bad={tb}  (ratio good:bad = {tg / max(tb,1):.1f}:1)")
    print(f"Train with:  python resnet/train_resnet.py --data datasets/pcb_patches "
          f"--size {min(patch, 256)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heal", action="store_true",
                    help="reconstruct clean plates (median of aligned copies) for GOOD")
    ap.add_argument("--build-plates-only", action="store_true",
                    help="build + cache clean plates and exit (for inspection)")
    ap.add_argument("--source", choices=list(SOURCE_PREFIX), default="hripcb")
    ap.add_argument("--patch", type=int, default=1024, help="patch size (px); bigger = less zoom")
    ap.add_argument("--stride", type=int, default=512, help="good-tile grid stride (px)")
    ap.add_argument("--good-per-board", type=int, default=10, help="good tiles per board (non-heal)")
    ap.add_argument("--good-per-plate", type=int, default=400,
                    help="good tiles per clean plate (heal): random translations across it")
    ap.add_argument("--aug", type=int, default=2, help="augmented variants per good tile")
    ap.add_argument("--bad-jitter", type=int, default=3, help="variants per defect")
    ap.add_argument("--shift-px", type=float, default=12.0)
    ap.add_argument("--rot-deg", type=float, default=12.0)
    ap.add_argument("--min-std", type=float, default=12.0,
                    help="min grayscale std for a GOOD tile (skip near-uniform blanks)")
    ap.add_argument("--defect-offset", type=float, default=0.0,
                    help="position augmentation: place each defect off-center by up to this "
                         "fraction of --patch (0 = centered, as before; try 0.3 for Goal 4)")
    ap.add_argument("--plate-n", type=int, default=25, help="copies to median per plate")
    ap.add_argument("--save-size", type=int, default=384,
                    help="downscale each patch to this before saving (0 = keep --patch). "
                         "The patch is cropped at --patch for zoom, then stored at this size; "
                         "keeps the dataset small since we train at 256.")
    ap.add_argument("--save-fmt", choices=["jpg", "png"], default="jpg",
                    help="jpg (small, q95) or png (lossless)")
    ap.add_argument("--max-good", type=int, default=60000)
    ap.add_argument("--max-bad", type=int, default=40000)
    ap.add_argument("--include-deeppcb", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng = random.Random(SEED)
    prefix = SOURCE_PREFIX[args.source]

    if args.build_plates_only:
        print("Building clean plates ...")
        build_clean_plates(prefix, args.plate_n, force=True)
        print(f"done -> {PLATES}")
        return

    if args.heal:
        print("Building clean plates (heal mode) ...")
        plates = build_clean_plates(prefix, args.plate_n)
        # GOOD from clean plates, BAD from defective images; group + split by template
        records = [("good", (p, [], f"tpl_{tid}")) for tid, p in plates.items()]
        for img, lab in _pku_images(prefix):
            m = _TPL.match(img.name)
            if m:
                records.append(("bad", (img, "yolo:" + str(lab), f"tpl_{m.group(1)}")))
        groups = sorted({r[1][2] for r in records})
        random.Random(SEED).shuffle(groups)
        # few templates -> hold out ~20% as val and ~20% as test (whole boards), e.g. 6/2/2 of 10
        n = len(groups)
        n_va = max(1, int(round(n * 0.2)))
        n_te = max(1, int(round(n * 0.2)))
        n_tr = n - n_va - n_te
        split_of = {g: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test")
                    for i, g in enumerate(groups)}
        print("template split:", {g: split_of[g][:2] for g in groups})
        args.good_per_board = args.good_per_plate
    else:
        boards = collect_pku(prefix) + (collect_deeppcb() if args.include_deeppcb else [])
        if not boards:
            print("No boards found."); return
        print(f"boards: {len(boards)}  (pku:{args.source}"
              + (" + deeppcb" if args.include_deeppcb else "") + ")")
        records = [("good", b) for b in boards] + [("bad", b) for b in boards]
        groups = sorted({b[2] for b in boards})
        random.Random(SEED).shuffle(groups)
        n = len(groups); n_tr, n_va = int(n * SPLIT[0]), int(n * SPLIT[1])
        split_of = {g: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test")
                    for i, g in enumerate(groups)}
        args.good_per_plate = args.good_per_board

    cap = {sp: {"good": int(args.max_good * r), "bad": int(args.max_bad * r)}
           for sp, r in zip(("train", "val", "test"), SPLIT)}
    counts = _materialize(Path(args.out), records, split_of, cap, args, rng)
    _report(counts, args.patch)


if __name__ == "__main__":
    main()
