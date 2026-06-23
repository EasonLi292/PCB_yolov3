#!/usr/bin/env python3
"""
Build consistent, ready-to-train YOLO datasets from the loaded PCB-defect sources.

All output datasets share the SAME layout, so they load and split identically:

    datasets/<name>/
        data.yaml
        train/images/  train/labels/
        val/images/    val/labels/
        test/images/   test/labels/

Outputs:
    unified_pku_yolo/   6-class PKU set  (norbertelter + HRIPCB + Roboflow merged)
    deeppcb_yolo/       6-class DeepPCB  (grayscale template/test pairs, converted)
    dspcbsd_yolo/       9-class DsPCBSD+ (broader surface taxonomy, relayouted)

Splitting is deterministic (SEED) with a single SPLIT ratio, and GROUP-AWARE: for the
PKU set, every augmented variant of the same original board/defect image is kept in the
same split (no train/val leakage). Re-running re-splits reproducibly.

Usage:
    python scripts/build_unified_yolo.py [pku|deeppcb|dspcbsd|all]
"""
import os, re, sys, shutil, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "datasets"

# ---- split config (edit these to re-split everything consistently) ----
SPLIT = (0.80, 0.10, 0.10)   # train, val, test
SEED = 42

# ---- canonical 6-class PKU taxonomy ----
CANON = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
CANON_IDX = {n: i for i, n in enumerate(CANON)}

# ---- DsPCBSD+ 9-class taxonomy (kept as-is) ----
DSPCBSD = ["SH", "SP", "SC", "OP", "MB", "HB", "CS", "CFO", "BMFO"]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)            # hardlink: no extra disk on same volume
    except OSError:
        shutil.copy2(src, dst)


def names_to_idxmap(names):
    """Map a source's local class indices -> canonical indices by (normalized) name."""
    return {i: CANON_IDX[nm.strip().lower()] for i, nm in enumerate(names)}


def remap_label_text(src_txt: Path, idx_map: dict) -> str:
    out = []
    if src_txt.exists():
        for line in src_txt.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            p = line.split()
            new = idx_map.get(int(float(p[0])))
            if new is None:
                continue
            out.append(" ".join([str(new)] + p[1:]))
    return "\n".join(out) + ("\n" if out else "")


# ---- group key for the PKU set: <board>_<defect>_<inst>, shared across augmentations ----
_CORE = re.compile(r"(\d+_(?:" + "|".join(CANON) + r")_\d+)", re.I)
def pku_group(stem: str) -> str:
    m = _CORE.search(stem)
    return m.group(1).lower() if m else stem


# A record is (src_img_path, label_text, group_key, out_filename)
def collect_pku():
    recs = []
    sources = [
        ("nb_", DS / "kaggle-pcb-defect" / "pcb-defect-dataset", ["train", "val", "test"],
         ["mouse_bite", "spur", "missing_hole", "short", "open_circuit", "spurious_copper"]),
        ("hr_", DS / "kaggle-hripcb" / "HRIPCB_UPDATE", ["train", "val", "test"],
         ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spurious_copper", "Spur"]),
        ("rf_", DS / "roboflow-pcb" / "v2", ["train", "valid", "test"],
         ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]),
    ]
    for prefix, root, splits, names in sources:
        idx_map = names_to_idxmap(names)
        for sp in splits:
            img_dir = root / sp / "images"
            lbl_dir = root / sp / "labels"
            if not img_dir.is_dir():
                continue
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in IMG_EXTS:
                    continue
                stem = img.stem
                txt = remap_label_text(lbl_dir / f"{stem}.txt", idx_map)
                recs.append((img, txt, pku_group(stem), f"{prefix}{stem}{img.suffix.lower()}"))
    return recs


def collect_deeppcb():
    pcb = DS / "deeppcb" / "PCBData"
    TYPE = {1: "open_circuit", 2: "short", 3: "mouse_bite",
            4: "spur", 5: "spurious_copper", 6: "missing_hole"}
    W = H = 640
    recs = []
    for listing in ("trainval.txt", "test.txt"):
        lf = pcb / listing
        if not lf.exists():
            continue
        for line in lf.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            img_rel, ann_rel = line.split()
            img = pcb / img_rel
            ann = pcb / ann_rel
            if not img.exists():
                cand = img.with_name(img.stem + "_test" + img.suffix)
                img = cand if cand.exists() else img
            if not img.exists() or not ann.exists():
                continue
            rows = []
            for l in ann.read_text().splitlines():
                p = l.split()
                if len(p) < 5:
                    continue
                x1, y1, x2, y2, t = map(int, map(float, p[:5]))
                cls = CANON_IDX[TYPE[t]]
                rows.append(f"{cls} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
                            f"{abs(x2-x1)/W:.6f} {abs(y2-y1)/H:.6f}")
            txt = "\n".join(rows) + ("\n" if rows else "")
            recs.append((img, txt, img.stem, f"{img.stem}{img.suffix}"))
    return recs


def collect_dspcbsd():
    root = DS / "kaggle-dspcbsd" / "DsPCBSD+" / "Data_YOLO"
    recs = []
    for sp in ("train", "val"):
        img_dir = root / "images" / sp
        lbl_dir = root / "labels" / sp
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lf = lbl_dir / f"{img.stem}.txt"
            txt = lf.read_text() if lf.exists() else ""
            recs.append((img, txt, img.stem, f"{img.stem}{img.suffix.lower()}"))
    return recs


def materialize(name, recs, class_names):
    out = DS / name
    if out.exists():
        shutil.rmtree(out)

    # group-aware deterministic split
    groups = {}
    for r in recs:
        groups.setdefault(r[2], []).append(r)
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    n = len(keys)
    n_tr = int(n * SPLIT[0])
    n_va = int(n * SPLIT[1])
    assign = {}
    for i, k in enumerate(keys):
        assign[k] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")

    counts = {"train": 0, "val": 0, "test": 0}
    for img, txt, grp, fname in recs:
        sp = assign[grp]
        link_or_copy(img, out / sp / "images" / fname)
        lbl = out / sp / "labels" / (Path(fname).stem + ".txt")
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text(txt)
        counts[sp] += 1

    yaml = ["path: " + str(out), "train: train/images", "val: val/images", "test: test/images",
            "", f"nc: {len(class_names)}", "names:"]
    for i, c in enumerate(class_names):
        yaml.append(f"  {i}: {c}")
    (out / "data.yaml").write_text("\n".join(yaml) + "\n")
    print(f"[{name}] {sum(counts.values())} images  ->  {counts}")


BUILDERS = {
    "pku":     (lambda: materialize("unified_pku_yolo", collect_pku(),     CANON)),
    "deeppcb": (lambda: materialize("deeppcb_yolo",     collect_deeppcb(), CANON)),
    "dspcbsd": (lambda: materialize("dspcbsd_yolo",     collect_dspcbsd(), DSPCBSD)),
}

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    targets = list(BUILDERS) if what == "all" else [what]
    for t in targets:
        BUILDERS[t]()
    print("done.")
