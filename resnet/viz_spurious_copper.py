#!/usr/bin/env python3
"""Show a spurious_copper defect and why it looks like GOOD to the 256/1024 classifier.

Panels: (1) full board with the defect boxed, (2) the defect zoomed in (clearly an extra
copper blob), (3) the actual BAD tile the model sees (1024 crop -> 256), (4) a clean GOOD
tile from the same board at the same scale. The point: zoomed in the defect is obvious, but
in the model's downscaled tile it's a few pixels of extra copper among lots of real copper.
"""
import glob, sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
SPUR_CU = 5   # class id


def boxes_of(img_path, W, H, cls):
    lf = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    out = []
    if Path(lf).exists():
        for ln in Path(lf).read_text().splitlines():
            q = ln.split()
            if len(q) >= 5 and int(float(q[0])) == cls:
                cx, cy, bw, bh = (float(v) for v in q[1:5])
                out.append((cx * W, cy * H, bw * W, bh * H))
    return out


def crop(src, cx, cy, sz):
    h, w = src.shape[:2]
    x0 = int(min(max(cx - sz / 2, 0), w - sz)); y0 = int(min(max(cy - sz / 2, 0), h - sz))
    return src[y0:y0 + sz, x0:x0 + sz], x0, y0


def clean_tile(src, boxes, sz, rng):
    """random 1024 window clear of every defect box (a GOOD tile)."""
    h, w = src.shape[:2]
    for _ in range(400):
        cx = rng.uniform(sz / 2, w - sz / 2); cy = rng.uniform(sz / 2, h - sz / 2)
        x0, y0 = int(cx - sz / 2), int(cy - sz / 2)
        ok = all(not (x0 - 40 < bx < x0 + sz + 40 and y0 - 40 < by < y0 + sz + 40)
                 for bx, by, _, _ in boxes)
        tile = src[y0:y0 + sz, x0:x0 + sz]
        if ok and tile.size and cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY).std() > 18:
            return tile
    return src[:sz, :sz]


def main():
    rng = np.random.default_rng(0)
    # pick the largest spurious_copper defect for visibility
    best = None
    for f in sorted(glob.glob(str(ROOT / "datasets/unified_pku_yolo/*/images/hr_*spurious_copper*"))):
        src = cv2.imread(f)
        if src is None:
            continue
        H, W = src.shape[:2]
        for cx, cy, bw, bh in boxes_of(f, W, H, SPUR_CU):
            if best is None or bw * bh > best[1]:
                best = ((f, src, cx, cy, bw, bh), bw * bh)
    if best is None:
        print("none found"); sys.exit(1)
    (f, src, cx, cy, bw, bh), _ = best
    H, W = src.shape[:2]
    allboxes = boxes_of(f, W, H, SPUR_CU)
    print(f"using {Path(f).name}  defect box {bw:.0f}x{bh:.0f}px on {W}x{H} board")

    P = 360   # display panel size
    # (1) full board, all spurious_copper boxed, the chosen one thicker
    board = src.copy()
    for bx, by, w2, h2 in allboxes:
        cv2.rectangle(board, (int(bx - w2/2), int(by - h2/2)), (int(bx + w2/2), int(by + h2/2)),
                      (0, 0, 255), 6)
    cv2.rectangle(board, (int(cx - bw/2)-10, int(cy - bh/2)-10),
                  (int(cx + bw/2)+10, int(cy + bh/2)+10), (0, 220, 255), 8)
    scale = P / max(H, W); board = cv2.resize(board, (int(W*scale), int(H*scale)))
    board = cv2.copyMakeBorder(board, 0, P - board.shape[0], 0, P - board.shape[1],
                               cv2.BORDER_CONSTANT, value=(30, 30, 30))

    # (2) zoomed defect (tight 256 crop, upscaled) with box
    zc, zx, zy = crop(src, cx, cy, 256)
    zoom = cv2.resize(zc, (P, P), interpolation=cv2.INTER_NEAREST)
    s = P / 256
    cv2.rectangle(zoom, (int((cx-zx-bw/2)*s), int((cy-zy-bh/2)*s)),
                  (int((cx-zx+bw/2)*s), int((cy-zy+bh/2)*s)), (0, 0, 255), 3)

    # (3) BAD tile the model sees: 1024 crop -> 256 -> upscaled for display
    bad1024, _, _ = crop(src, cx, cy, 1024)
    bad256 = cv2.resize(bad1024, (256, 256), interpolation=cv2.INTER_AREA)
    bad = cv2.resize(bad256, (P, P), interpolation=cv2.INTER_NEAREST)

    # (4) GOOD tile from same board, same 1024 -> 256 pipeline
    good1024 = clean_tile(src, allboxes, 1024, rng)
    good256 = cv2.resize(good1024, (256, 256), interpolation=cv2.INTER_AREA)
    good = cv2.resize(good256, (P, P), interpolation=cv2.INTER_NEAREST)

    labels = ["1) full board (defect boxed)", "2) defect zoomed in",
              "3) BAD tile @256 (model view)", "4) GOOD tile @256 (same board)"]
    panels = []
    for img, lab in zip([board, zoom, bad, good], labels):
        bar = np.full((40, P, 3), 40, np.uint8)
        cv2.putText(bar, lab, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        panels.append(np.vstack([bar, img]))
    gap = np.full((panels[0].shape[0], 10, 3), 255, np.uint8)
    row = panels[0]
    for p in panels[1:]:
        row = np.hstack([row, gap, p])
    title = np.full((46, row.shape[1], 3), 255, np.uint8)
    cv2.putText(title, "spurious_copper: obvious zoomed in (2), but a few extra copper pixels "
                "among real copper once downscaled (3) -> looks like GOOD (4)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    out = np.vstack([title, row])
    op = ROOT / "resnet/figures/spurious_copper_why.png"
    op.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(op), out)
    print("saved ->", op)


if __name__ == "__main__":
    main()
