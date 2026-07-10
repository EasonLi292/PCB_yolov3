#!/usr/bin/env python3
"""Visualize what --defect-offset does: crop a real defect at 0,10,20,30,40% off-center.

`mine_patches.py --defect-offset 0.4` draws the defect's displacement from the tile
center UNIFORMLY in [-0.4*patch, +0.4*patch] on EACH axis -- so a bad tile can have the
defect anywhere from dead-center out to 40% of the tile toward any corner. This renders a
single real defect at fixed offsets along the diagonal so the progression is easy to see.
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
CLS = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]


def boxes_of(img_path, W, H):
    lf = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    out = []
    if Path(lf).exists():
        for ln in Path(lf).read_text().splitlines():
            q = ln.split()
            if len(q) >= 5:
                cid = int(float(q[0])); cx, cy, bw, bh = (float(v) for v in q[1:5])
                out.append((cid, cx * W, cy * H, bw * W, bh * H))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=1024)
    ap.add_argument("--panel", type=int, default=320, help="display size per panel")
    ap.add_argument("--prefer", default="missing_hole", help="defect class to prefer")
    ap.add_argument("--out", default=str(ROOT / "resnet/figures/offset_increments.png"))
    args = ap.parse_args()
    fracs = [0.0, 0.1, 0.2, 0.3, 0.4]
    half = args.patch // 2
    room = half + int(0.45 * args.patch)
    prefer_id = CLS.index(args.prefer)

    # find a clear defect with enough in-bounds room to shift the crop to 40%
    pick = None
    for f in sorted(glob.glob(str(ROOT / "datasets/unified_pku_yolo/*/images/hr_*"))):
        src = cv2.imread(f)
        if src is None:
            continue
        H, W = src.shape[:2]
        for cid, dcx, dcy, bw, bh in boxes_of(f, W, H):
            if cid == prefer_id and room < dcx < W - room and room < dcy < H - room:
                pick = (f, src, cid, dcx, dcy, bw, bh)
                break
        if pick:
            break
    if not pick:
        print("no suitable defect found"); sys.exit(1)
    f, src, cid, dcx, dcy, bw, bh = pick
    print(f"using {Path(f).name}  class={CLS[cid]}  box=({dcx:.0f},{dcy:.0f},{bw:.0f}x{bh:.0f})")

    panels = []
    for fr in fracs:
        off = int(fr * args.patch)
        # shift the crop diagonally so the defect moves toward the top-left corner
        ox, oy = int(dcx - half + off), int(dcy - half + off)
        crop = src[oy:oy + args.patch, ox:ox + args.patch].copy()
        # defect box position inside this crop
        bx, by = dcx - ox, dcy - oy
        s = args.panel / args.patch
        disp = cv2.resize(crop, (args.panel, args.panel), interpolation=cv2.INTER_AREA)
        # red box on the defect, yellow crosshair at tile center
        x1, y1 = int((bx - bw / 2) * s), int((by - bh / 2) * s)
        x2, y2 = int((bx + bw / 2) * s), int((by + bh / 2) * s)
        pad = 6
        cv2.rectangle(disp, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), (0, 0, 255), 3)
        c = args.panel // 2
        cv2.drawMarker(disp, (c, c), (0, 220, 255), cv2.MARKER_CROSS, 26, 2)
        # label bar
        bar = np.full((46, args.panel, 3), 40, np.uint8)
        cv2.putText(bar, f"{int(fr*100)}% off-center", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        panels.append(np.vstack([bar, disp]))

    gap = np.full((panels[0].shape[0], 12, 3), 255, np.uint8)
    row = panels[0]
    for p in panels[1:]:
        row = np.hstack([row, gap, p])
    title = np.full((54, row.shape[1], 3), 255, np.uint8)
    cv2.putText(title, f"--defect-offset 0.4 : {CLS[cid]} placed 0-40% off tile center "
                f"(yellow=center, red=defect)", (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    out = np.vstack([title, row])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, out)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
