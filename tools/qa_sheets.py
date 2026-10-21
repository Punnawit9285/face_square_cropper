"""Contact sheets of the outputs with a crosshair through each image center, for visual QA.

usage: qa_sheets.py <out_dir> <sheet_dir> [tile_px] [cols] [rows] [names_file]

names_file (optional): comma-separated file names to include instead of every file in out_dir.
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    out_dir, sheet_dir = Path(sys.argv[1]), Path(sys.argv[2])
    tile = int(sys.argv[3]) if len(sys.argv) > 3 else 240
    cols = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    rows = int(sys.argv[5]) if len(sys.argv) > 5 else 6
    label = 16
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 11)
    except OSError:
        font = ImageFont.load_default()
    names = sorted(p.name for p in out_dir.iterdir() if not p.name.startswith("."))
    if len(sys.argv) > 6:
        wanted = set(Path(sys.argv[6]).read_text().strip().split(","))
        names = [n for n in names if n in wanted]
    sheet_dir.mkdir(parents=True, exist_ok=True)
    per = cols * rows
    for si in range(0, len(names), per):
        sheet = Image.new("RGB", (cols * tile, rows * (tile + label)), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        for k, name in enumerate(names[si:si + per]):
            im = Image.open(out_dir / name)
            w, h = im.size
            im = im.convert("RGBA")
            im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im).convert("RGB")
            im = im.resize((tile, tile), Image.LANCZOS)
            x, y = (k % cols) * tile, (k // cols) * (tile + label)
            sheet.paste(im, (x, y))
            c = tile / 2
            d.line([(x + c, y), (x + c, y + tile - 1)], fill=(255, 0, 0), width=1)
            d.line([(x, y + c), (x + tile - 1, y + c)], fill=(255, 0, 0), width=1)
            d.rectangle([x, y, x + tile - 1, y + tile - 1], outline=(200, 200, 200))
            d.text((x + 3, y + tile + 2), f"{si + k + 1}. {name[:13]} {w}x{h}", fill=(0, 0, 0), font=font)
        sheet.save(sheet_dir / f"qa_{si // per + 1:02d}.jpg", quality=88)
    print(f"qa: {len(names)} images on {-(-len(names) // per)} sheets -> {sheet_dir}")


if __name__ == "__main__":
    main()
