"""Review sheets for detector disagreements: face region with the MediaPipe face oval and all candidate centers.

usage: review_sheets.py <image_dir> <detect.json> <centers.json> <out_dir> [flagged|all|file1,file2,...] [tile] [cols]

Green outline/plus = MediaPipe face oval and its box center, red circle = Vision (bias-corrected),
blue x = YuNet (bias-corrected), magenta diamond = final center when it differs from the oval center
(visible-face shift or manual override).
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROWS, LABEL = 3, 34


def font(size):
    for p in ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def tile_image(image_dir, e, c, tile):
    im = Image.open(image_dir / e["file"]).convert("RGBA")
    im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im).convert("RGB")
    side = 2.4 * c["face_h"]
    x0, y0 = round(c["cx"] - side / 2), round(c["cy"] - side / 2)
    s = round(side)
    region = im.crop((x0, y0, x0 + s, y0 + s)).resize((tile, tile), Image.LANCZOS)  # out of bounds -> black
    d = ImageDraw.Draw(region)
    to = lambda p: ((p[0] - x0) * tile / s, (p[1] - y0) * tile / s)
    mp = e["mp"]
    if "oval" in mp:
        pts = [to(p) for p in mp["oval"]]
        d.line(pts + [pts[0]], fill=(0, 255, 0), width=1)
        d.rectangle([to(mp["box"][:2]), to(mp["box"][2:])], outline=(0, 160, 0), width=1)
        g = to(mp["center"])
        d.line([(g[0] - 9, g[1]), (g[0] + 9, g[1])], fill=(0, 255, 0), width=2)
        d.line([(g[0], g[1] - 9), (g[0], g[1] + 9)], fill=(0, 255, 0), width=2)
    if "vision" in c["corrected"]:
        p = to(c["corrected"]["vision"])
        d.ellipse([p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6], outline=(255, 40, 40), width=2)
    if "yunet" in c["corrected"]:
        p = to(c["corrected"]["yunet"])
        d.line([(p[0] - 6, p[1] - 6), (p[0] + 6, p[1] + 6)], fill=(40, 120, 255), width=2)
        d.line([(p[0] - 6, p[1] + 6), (p[0] + 6, p[1] - 6)], fill=(40, 120, 255), width=2)
    if c["source"] != "mediapipe_oval":
        p = to((c["cx"], c["cy"]))
        d.polygon([(p[0], p[1] - 8), (p[0] + 8, p[1]), (p[0], p[1] + 8), (p[0] - 8, p[1])], outline=(255, 0, 255))
    return region


def main():
    image_dir, det_path, centers_path, out_dir = (Path(a) for a in sys.argv[1:5])
    which = sys.argv[5] if len(sys.argv) > 5 else "flagged"
    tile = int(sys.argv[6]) if len(sys.argv) > 6 else 320
    cols = int(sys.argv[7]) if len(sys.argv) > 7 else 4
    det = {e["file"]: e for e in json.loads(det_path.read_text())}
    centers = json.loads(centers_path.read_text())
    if which == "flagged":
        names = [n for n, c in centers.items() if c["flags"]]
    elif which == "all":
        names = list(centers)
    else:
        names = which.split(",")
    out_dir.mkdir(parents=True, exist_ok=True)
    f_small = font(11)
    per = cols * ROWS
    for si in range(0, len(names), per):
        sheet = Image.new("RGB", (cols * tile, ROWS * (tile + LABEL)), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        for k, name in enumerate(names[si:si + per]):
            c = centers[name]
            x, y = (k % cols) * tile, (k // cols) * (tile + LABEL)
            sheet.paste(tile_image(image_dir, det[name], c, tile), (x, y))
            r = " ".join(f"{a[0]}={v:.3f}" for a, v in c["resid"].items())
            d.text((x + 3, y + tile + 2), f"{si + k + 1}. {name[:13]}  {c['w']}x{c['h']}  {r}", fill=(0, 0, 0), font=f_small)
            d.text((x + 3, y + tile + 17), ",".join(c["flags"])[:52], fill=(170, 0, 0), font=f_small)
        out = out_dir / f"review_{si // per + 1:02d}.jpg"
        sheet.save(out, quality=90)
        print(out)


if __name__ == "__main__":
    main()
