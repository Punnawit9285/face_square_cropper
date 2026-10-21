"""Show, step by step, how one photo is cropped. Run run_all.sh first: this reads its results in work/.

usage: explain_one.py <input_dir> <output_dir> <file_name> <out.png>

Panel 1  Find the face: Apple Vision box (red), YuNet box (blue), MediaPipe face outline (green).
Panel 2  Face center: middle of the face outline (green +). If hair hides part of the face, the visible
         face skin (tinted yellow) moves the center to the middle of what you can see (magenta diamond).
Panel 3  Largest square: distance from the center to each photo edge (the shortest, in yellow, sets the
         square), frames excluded (shaded red), and the crop box.
Panel 4  Result: the saved square with a crosshair through its center.
"""
import csv
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import step3_visible_face as visible_face  # noqa: E402  (reuses the pipeline's own visible-face measurement)

PANEL, TITLE, CAPTION, GAP, HEADER = 520, 34, 96, 16, 46
GREEN, RED, BLUE, MAGENTA, YELLOW, GRAY = (0, 220, 0), (255, 50, 50), (40, 120, 255), (255, 0, 255), (255, 205, 0), (170, 170, 170)


def font(size, bold=False):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size, index=1 if bold else 0)
    except OSError:
        return ImageFont.load_default()


def wrap(text, f, width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if f.getlength(trial) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + [line]


def crop_padded(im, box, fill=(235, 235, 235)):
    x0, y0, x1, y1 = (round(v) for v in box)
    canvas = Image.new("RGB", (x1 - x0, y1 - y0), fill)
    canvas.paste(im, (-x0, -y0))
    return canvas


class View:
    """Shows the source region (x0, y0, w, h) letterboxed in a PANEL x PANEL square; maps source px to panel px."""

    def __init__(self, im, x0, y0, w, h, overlay=None):
        self.s = PANEL / max(w, h)
        self.ox, self.oy = (PANEL - w * self.s) / 2 - x0 * self.s, (PANEL - h * self.s) / 2 - y0 * self.s
        region = crop_padded(im, (x0, y0, x0 + w, y0 + h))
        if overlay is not None:
            region = Image.alpha_composite(region.convert("RGBA"), overlay).convert("RGB")
        self.panel = Image.new("RGB", (PANEL, PANEL), (235, 235, 235))
        self.panel.paste(region.resize((max(1, round(w * self.s)), max(1, round(h * self.s))), Image.LANCZOS),
                         (round(self.ox + x0 * self.s), round(self.oy + y0 * self.s)))
        self.draw = ImageDraw.Draw(self.panel, "RGBA")

    def __call__(self, x, y):
        return x * self.s + self.ox, y * self.s + self.oy

    def rect(self, x0, y0, x1, y1, color, width=3, fill=None):
        self.draw.rectangle([self(x0, y0), self(x1, y1)], outline=color, width=width, fill=fill)

    def plus(self, x, y, color, r=11):
        px, py = self(x, y)
        self.draw.line([(px - r, py), (px + r, py)], fill=color, width=3)
        self.draw.line([(px, py - r), (px, py + r)], fill=color, width=3)

    def diamond(self, x, y, color, r=10):
        px, py = self(x, y)
        self.draw.polygon([(px, py - r), (px + r, py), (px, py + r), (px - r, py)], outline=color, width=3)

    def ring(self, x, y, color, r=7):
        px, py = self(x, y)
        self.draw.ellipse([px - r, py - r, px + r, py + r], outline=color, width=2)

    def cross(self, x, y, color, r=7):
        px, py = self(x, y)
        self.draw.line([(px - r, py - r), (px + r, py + r)], fill=color, width=2)
        self.draw.line([(px - r, py + r), (px + r, py - r)], fill=color, width=2)


def load(name):
    work = ROOT / "work"
    det = next(e for e in json.loads((work / "detect.json").read_text()) if e["file"] == name)
    center = json.loads((work / "centers.json").read_text())[name]
    vis = json.loads((work / "visible.json").read_text())[name]
    border = json.loads((work / "borders.json").read_text())[name]
    row = next(r for r in csv.DictReader((work / "manifest.csv").open()) if r["file"] == name)
    return det, center, vis, border, row


def main():
    in_dir, out_dir, name, out_png = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    det, center, vis, border, row = load(name)
    src = Image.open(in_dir / name).convert("RGBA")
    im = Image.alpha_composite(Image.new("RGBA", src.size, (255, 255, 255, 255)), src).convert("RGB")
    W, H = im.size
    m = det["mp"]
    cx, cy = float(row["face_cx"]), float(row["face_cy"])  # final (shared) center actually used for the crop
    side, x0, y0 = int(row["side"]), int(row["x0"]), int(row["y0"])
    trim = dict(zip(("top", "bottom", "left", "right"), map(int, row["frame_trim_tblr"].split("/"))))
    panels = []

    # 1. find the face
    v = View(im, 0, 0, W, H)
    vf = det["vision"]["faces"][0]
    v.rect(vf["x"], vf["y"], vf["x"] + vf["w"], vf["y"] + vf["h"], RED)
    if det["yunet"]["match"]:
        y = det["yunet"]["match"]
        v.rect(y["x"], y["y"], y["x"] + y["w"], y["y"] + y["h"], BLUE)
    v.draw.line([v(*p) for p in m["oval"]] + [v(*m["oval"][0])], fill=GREEN, width=3)
    n = max(det["vision"]["n"], det["yunet"]["n"])
    extra = f" {n} faces were found; the largest one is the subject." if n > 1 else ""
    panels.append((v.panel, "1. Find the face",
                   "Apple Vision (red) and YuNet (blue) locate the face; MediaPipe's 478-point face mesh traces "
                   f"its outline (green).{extra}"))

    # 2. decide the face center
    zs = 2.6 * m["height"]
    zx0, zy0 = cx - zs / 2, cy - zs / 2
    overlay, shifted = None, center["source"].endswith("visible_shift")
    if shifted:
        dbg = visible_face.measure(visible_face.create_segmenter(
            ROOT / "models/segmenter/selfie_multiclass_256x256.tflite"),
            visible_face.load_rgb(in_dir / name), m, debug=True)["debug"]
        tint = Image.new("RGBA", (dbg["side"], dbg["side"]), (255, 220, 0, 110))
        mask = Image.fromarray(dbg["visible"].astype("uint8") * 255).resize((dbg["side"], dbg["side"]), Image.NEAREST)
        overlay = Image.new("RGBA", (round(zx0 + zs) - round(zx0), round(zy0 + zs) - round(zy0)), (0, 0, 0, 0))
        overlay.paste(tint, (dbg["x0"] - round(zx0), dbg["y0"] - round(zy0)), mask)
    v = View(im, zx0, zy0, zs, zs, overlay)
    v.draw.line([v(*p) for p in m["oval"]] + [v(*m["oval"][0])], fill=GREEN, width=2)
    v.rect(*m["box"], (0, 150, 0), width=1)
    for k, mark in (("vision", v.ring), ("yunet", v.cross)):
        if k in center["corrected"]:
            mark(*center["corrected"][k], RED if k == "vision" else BLUE)
    v.plus(*m["center"], GREEN)
    checks = [f"{label} {center['resid'][k] * 100:.0f}%" for k, label in (("yunet", "YuNet (blue x)"),
              ("vision", "Apple Vision (red ring)")) if k in center["resid"]]
    text = ("The face center is the middle of the face outline (green +). Cross-check, distance in face heights: "
            + ", ".join(checks) + ".")
    if shifted:
        v.draw.line([v(*m["center"]), v(cx, cy)], fill=MAGENTA, width=2)
        v.diamond(cx, cy, MAGENTA)
        text = (f"Middle of the face outline = green +. Hair hides {100 * (1 - vis['width_cover']):.0f}% of the face "
                f"width, so the center moves {vis['vis_shift']:+.0f}px to the middle of the visible face "
                f"(tinted): magenta diamond.")
    panels.append((v.panel, "2. Decide the face center", text))

    # 3. largest centered square inside the photo area
    v = View(im, 0, 0, W, H)
    area = (trim["left"], trim["top"], W - trim["right"], H - trim["bottom"])
    shade = (255, 0, 0, 90)
    for box in ((0, 0, W, area[1]), (0, area[3], W, H), (0, 0, area[0], H), (area[2], 0, W, H)):
        if box[2] > box[0] and box[3] > box[1]:
            v.rect(*box, None, width=0, fill=shade)
    dist = {"left": cx - area[0], "right": area[2] - cx, "top": cy - area[1], "bottom": area[3] - cy}
    limit = min(dist, key=dist.get)
    ends = {"left": (area[0], cy), "right": (area[2], cy), "top": (cx, area[1]), "bottom": (cx, area[3])}
    for k, end in ends.items():
        v.draw.line([v(cx, cy), v(*end)], fill=YELLOW if k == limit else GRAY, width=5 if k == limit else 2)
    v.rect(x0, y0, x0 + side, y0 + side, YELLOW, width=3)
    v.plus(cx, cy, MAGENTA if shifted else GREEN)
    framed = any(trim.values())
    text = (f"The shortest distance from the face center to a photo edge ({limit}, {dist[limit]:.0f}px, yellow) "
            f"sets the square: side = 2 x {dist[limit]:.0f} -> {side}px.")
    if framed:
        text += " Frame excluded (red): " + ", ".join(f"{k} {t}px" for k, t in trim.items() if t) + "."
    panels.append((v.panel, "3. Fit the largest centered square", text))

    # 4. result
    out = Image.open(out_dir / name).convert("RGBA")
    out = Image.alpha_composite(Image.new("RGBA", out.size, (255, 255, 255, 255)), out).convert("RGB")
    v = View(out, 0, 0, out.width, out.height)
    v.draw.line([v(out.width / 2, 0), v(out.width / 2, out.height)], fill=(255, 0, 0), width=1)
    v.draw.line([v(0, out.height / 2), v(out.width, out.height / 2)], fill=(255, 0, 0), width=1)
    err = max(abs(float(row["center_err_x"])), abs(float(row["center_err_y"])))
    panels.append((v.panel, "4. Save the square",
                   f"{side}x{side}px {row['format']}, cut straight from the original (no resizing, no padding). "
                   f"Crop center is {err:.1f}px from the face center (pixel rounding)."))

    f_title, f_cap, f_head = font(19, bold=True), font(15), font(17, bold=True)
    cell_h = TITLE + PANEL + CAPTION
    sheet = Image.new("RGB", (2 * PANEL + 3 * GAP, HEADER + 2 * cell_h + GAP), "white")
    d = ImageDraw.Draw(sheet)
    d.text((GAP, 14), f"{name}   ({W}x{H} {row['format']})", fill=(0, 0, 0), font=f_head)
    for i, (panel, title, caption) in enumerate(panels):
        x, y = GAP + (i % 2) * (PANEL + GAP), HEADER + (i // 2) * (cell_h + GAP // 2)
        d.text((x, y + 6), title, fill=(0, 0, 0), font=f_title)
        sheet.paste(panel, (x, y + TITLE))
        for j, line in enumerate(wrap(caption, f_cap, PANEL)):
            d.text((x, y + TITLE + PANEL + 6 + j * 19), line, fill=(60, 60, 60), font=f_cap)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    print(out_png)


if __name__ == "__main__":
    main()
