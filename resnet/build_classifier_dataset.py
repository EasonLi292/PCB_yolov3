#!/usr/bin/env python3
"""
Assemble a good-vs-defective board-classification dataset for the ResNet classifier.

Output (same train/val/test discipline as the detector, group-aware + deterministic):

    datasets/pcb_goodbad/
        train/good/  train/bad/
        val/good/    val/bad/
        test/good/   test/bad/

Sources of GOOD (defect-free) boards:
  * DeepPCB *_temp.jpg  -- the registered reference image paired with every test
    board; these are genuinely defect-free.                       (~1500 locally)
  * Any image you drop under  resnet/extra_good/**  -- this is where downloaded
    "good PCB" datasets go (see resnet/README.md for sources). Recursively scanned.

Sources of BAD (defective) boards:
  * DeepPCB *_test.jpg  -- the tested board carrying the defects.  (~1500 locally)
  * Optional: --pku-bad N  sampled boards from datasets/unified_pku_yolo (every
    image there has photoshopped defects, so all are "bad").

Because GOOD boards are the scarce, valuable class (we want many more of them than
BAD), keep adding to extra_good/ and re-run. Splitting is group-aware: a DeepPCB
board id never appears in two splits (its temp+test pair stays together), so the
model can't memorize a board's texture across the train/test boundary.

Usage:
    python resnet/build_classifier_dataset.py                 # DeepPCB + extra_good
    python resnet/build_classifier_dataset.py --pku-bad 1500  # add 1500 PKU bads
"""
import argparse, random, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "datasets"
EXTRA_GOOD = Path(__file__).resolve().parent / "extra_good"
OUT = DS / "pcb_goodbad"

SPLIT = (0.80, 0.10, 0.10)   # train / val / test  (matches the detector)
SEED = 42
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        import os
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# A record = (src_path, label("good"|"bad"), group_key, out_filename)
def collect_deeppcb():
    recs = []
    base = DS / "deeppcb" / "PCBData"
    if not base.is_dir():
        return recs
    for p in base.rglob("*.jpg"):
        stem = p.stem                      # e.g. 00041000_temp / 00041000_test
        if stem.endswith("_temp"):
            label, board = "good", stem[:-5]
        elif stem.endswith("_test"):
            label, board = "bad", stem[:-5]
        else:
            continue
        recs.append((p, label, f"dp_{board}", f"dp_{stem}{p.suffix}"))
    return recs


def collect_extra_good():
    recs = []
    if not EXTRA_GOOD.is_dir():
        return recs
    for p in sorted(EXTRA_GOOD.rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            # group by parent folder so a multi-shot board stays in one split
            recs.append((p, "good", f"xg_{p.parent.name}_{p.stem}", f"xg_{p.stem}{p.suffix}"))
    return recs


def collect_pku_bad(limit):
    recs = []
    src = DS / "unified_pku_yolo"
    if not src.is_dir() or limit <= 0:
        return recs
    imgs = []
    for sp in ("train", "val", "test"):
        d = src / sp / "images"
        if d.is_dir():
            imgs += sorted(d.iterdir())
    imgs = [p for p in imgs if p.suffix.lower() in IMG_EXTS]
    random.Random(SEED).shuffle(imgs)
    for p in imgs[:limit]:
        recs.append((p, "bad", f"pku_{p.stem}", f"pku_{p.stem}{p.suffix}"))
    return recs


def materialize(recs):
    if OUT.exists():
        shutil.rmtree(OUT)
    # group-aware deterministic split
    groups = {}
    for r in recs:
        groups.setdefault(r[2], []).append(r)
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    n = len(keys)
    n_tr, n_va = int(n * SPLIT[0]), int(n * SPLIT[1])
    assign = {k: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test")
              for i, k in enumerate(keys)}

    counts = {sp: {"good": 0, "bad": 0} for sp in ("train", "val", "test")}
    for src, label, grp, fname in recs:
        sp = assign[grp]
        link_or_copy(src, OUT / sp / label / fname)
        counts[sp][label] += 1

    print(f"[pcb_goodbad] -> {OUT}")
    for sp in ("train", "val", "test"):
        g, b = counts[sp]["good"], counts[sp]["bad"]
        print(f"  {sp:5s}: good={g:5d}  bad={b:5d}  total={g + b}")
    tot_g = sum(counts[sp]["good"] for sp in counts)
    tot_b = sum(counts[sp]["bad"] for sp in counts)
    print(f"  TOTAL: good={tot_g}  bad={tot_b}")
    if tot_g < tot_b:
        print("  NOTE: fewer GOOD than BAD. Drop more good boards in resnet/extra_good/ "
              "(see resnet/README.md) and re-run to grow the good class.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pku-bad", type=int, default=0,
                    help="also pull N defective boards from unified_pku_yolo as 'bad'")
    args = ap.parse_args()

    recs = collect_deeppcb() + collect_extra_good() + collect_pku_bad(args.pku_bad)
    if not recs:
        print("No source images found. Is datasets/deeppcb present?")
        return
    ng = sum(1 for r in recs if r[1] == "good")
    nb = sum(1 for r in recs if r[1] == "bad")
    print(f"collected: good={ng}  bad={nb}  (deeppcb + extra_good"
          + (f" + {args.pku_bad} pku" if args.pku_bad else "") + ")")
    materialize(recs)
    print("done.")


if __name__ == "__main__":
    main()
