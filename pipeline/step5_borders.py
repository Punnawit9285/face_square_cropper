"""Detect frames/bars along image edges (white mats, black/gray rules, drop shadows, letterbox bars).

usage: step5_borders.py <image_dir> <out.json>

For each side, scan lines inward up to MAX_DEPTH of the image size. A frame is a run of flat lines (low spread
along the line) that is colorless overall (white/black/gray: low median chroma across the run; judged per run,
not per line, because JPEG chroma subsampling tints the last mat lines next to the photo) and ends in a straight
edge where at least half of the line changes at once. The border width is the innermost such edge.
Colored design elements (e.g. a navy poster band) are not frames. "trim" adds a 1 px safety margin so JPEG blur
along the frame line cannot leave a faint stripe in the crop.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

MAX_DEPTH = 0.10   # fraction of the image dimension searched for a frame
FLAT_STD = 14.0    # max luminance spread along a line for it to count as frame
CHROMA_MAX = 30.0  # max median line chroma (max(R,G,B) - min(R,G,B)) over a frame run: frames are white/black/gray
EDGE_MIN = 18.0    # min median |luminance change| between consecutive lines for a straight full-length edge
SIDES = ("top", "bottom", "left", "right")


def side_view(a, side):
    """Lines ordered from the outer edge inward, each restricted to the central 80% of its length."""
    v = {"top": a, "bottom": a[::-1], "left": a.swapaxes(0, 1), "right": a.swapaxes(0, 1)[::-1]}[side]
    n = v.shape[1]
    return v[:, int(n * 0.1):int(n * 0.9)]


def border_width(lum, chroma, depth):
    flat = lum[:depth].std(axis=1) <= FLAT_STD
    line_chroma = chroma[:depth].mean(axis=1)
    edges = np.median(np.abs(np.diff(lum[:depth + 1], axis=0)), axis=1)  # edges[k-1]: change from line k-1 to k
    width = 0
    for k in range(1, depth + 1):
        if not flat[k - 1]:
            break  # line k-1 is photo content, so nothing at or beyond it can be frame
        if edges[k - 1] >= EDGE_MIN and np.median(line_chroma[:k]) <= CHROMA_MAX:
            width = k
    return width


def detect(path):
    """{"border": px per side, "trim": px per side to remove (border + 1 px margin, or 0)}."""
    im = Image.open(path).convert("RGBA")
    rgb = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im).convert("RGB")
    a = np.asarray(rgb, dtype=np.float64)
    lum = np.asarray(rgb.convert("L"), dtype=np.float64)
    chroma = a.max(axis=2) - a.min(axis=2)
    h, w = lum.shape
    border = {}
    for side in SIDES:
        depth = max(2, int(round(MAX_DEPTH * (h if side in ("top", "bottom") else w))))
        border[side] = border_width(side_view(lum, side), side_view(chroma, side), depth)
    return {"border": border, "trim": {s: v + 1 if v else 0 for s, v in border.items()}}


def main():
    image_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    out = {p.name: detect(p) for p in sorted(image_dir.iterdir()) if not p.name.startswith(".")}
    out_path.write_text(json.dumps(out, indent=1))
    framed = sum(any(v["border"].values()) for v in out.values())
    print(f"borders: {len(out)} images, {framed} with a frame/bar to exclude -> {out_path}")


if __name__ == "__main__":
    main()
