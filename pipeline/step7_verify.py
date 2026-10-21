"""Structural verification of the cropped outputs against their sources.

usage: step7_verify.py <src_dir> <out_dir> <manifest.csv>

Checks per file: square with the manifest side; same format and mode; ICC profile bytes identical; DPI kept;
JPEG has no EXIF thumbnail (IFD1), no XMP, no Photoshop/IPTC block, EXIF pixel dimensions updated;
pixels match the source crop box (PNG exact, JPEG PSNR >= 40 dB) and every 1-px shifted box matches worse.
"""
import csv
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

MIN_PSNR = 40.0


def jpeg_segments(data):
    i, segs = 2, []
    while i < len(data) - 4 and data[i] == 0xFF:
        m = data[i + 1]
        if m == 0xD8 or 0xD0 <= m <= 0xD7 or m == 0x01:
            i += 2
            continue
        ln = struct.unpack(">H", data[i + 2:i + 4])[0]
        segs.append((m, data[i + 4:i + 2 + ln]))
        if m == 0xDA:
            break
        i += 2 + ln
    return segs


def has_exif_thumbnail(seg):
    t = seg[6:]
    bo = "<" if t[:2] == b"II" else ">"
    off0 = struct.unpack(bo + "I", t[4:8])[0]
    n0 = struct.unpack(bo + "H", t[off0:off0 + 2])[0]
    next_ifd = struct.unpack(bo + "I", t[off0 + 2 + 12 * n0:off0 + 6 + 12 * n0])[0]
    return next_ifd != 0


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    src_dir, out_dir, manifest = (Path(a) for a in sys.argv[1:4])
    rows = list(csv.DictReader(manifest.open()))
    problems, psnrs = [], []
    src_names = sorted(p.name for p in src_dir.iterdir() if not p.name.startswith("."))
    out_names = sorted(p.name for p in out_dir.iterdir() if not p.name.startswith("."))
    if src_names != out_names:
        problems.append(f"name sets differ: missing {set(src_names) - set(out_names)}, extra {set(out_names) - set(src_names)}")
    for r in rows:
        name, side, x0, y0 = r["file"], int(r["side"]), int(r["x0"]), int(r["y0"])
        src, out = Image.open(src_dir / name), Image.open(out_dir / name)
        bad = lambda msg: problems.append(f"{name}: {msg}")
        if out.size != (side, side):
            bad(f"size {out.size} != {side}x{side}")
        if out.format != src.format:
            bad(f"format {out.format} != {src.format}")
        if out.mode != src.mode:
            bad(f"mode {out.mode} != {src.mode}")
        if out.info.get("icc_profile") != src.info.get("icc_profile"):
            bad("ICC profile differs")
        if src.info.get("dpi") and tuple(round(v) for v in out.info.get("dpi", (0, 0))) != tuple(round(v) for v in src.info["dpi"]):
            bad(f"dpi {out.info.get('dpi')} != {src.info['dpi']}")
        if out.format == "JPEG":
            for m, seg in jpeg_segments((out_dir / name).read_bytes()):
                if m == 0xE1 and seg.startswith(b"Exif\x00\x00") and has_exif_thumbnail(seg):
                    bad("EXIF thumbnail (IFD1) present")
                if m == 0xE1 and b"ns.adobe.com/xap" in seg[:40]:
                    bad("XMP present")
                if m == 0xED:
                    bad("Photoshop/IPTC block present")
            sub = out.getexif().get_ifd(0x8769)
            for tag in (0xA002, 0xA003):
                if tag in sub and sub[tag] != side:
                    bad(f"EXIF pixel dimension tag {tag:#x} = {sub[tag]}")
            if out.getexif().get(0x0112, 1) != 1:
                bad("EXIF orientation != 1")

        a = np.asarray(out.convert("RGBA"))
        full = np.asarray(src.convert("RGBA"))
        ref = full[y0:y0 + side, x0:x0 + side]
        if out.format == "PNG":
            if not np.array_equal(a, ref):
                bad("PNG pixels differ from source crop")
        else:
            p0 = psnr(a[..., :3], ref[..., :3])
            psnrs.append(p0)
            if p0 < MIN_PSNR:
                bad(f"PSNR {p0:.1f} dB < {MIN_PSNR}")
        # every 1-px shifted crop box must match worse than the chosen one (proves no off-by-one / shift)
        base = psnr(a[..., :3], ref[..., :3])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (dx, dy) == (0, 0):
                    continue
                xx, yy = x0 + dx, y0 + dy
                if xx < 0 or yy < 0 or xx + side > full.shape[1] or yy + side > full.shape[0]:
                    continue
                if psnr(a[..., :3], full[yy:yy + side, xx:xx + side, :3]) >= base:
                    bad(f"shifted box ({dx},{dy}) matches as well as the chosen box")
    print(f"verify: {len(rows)} files checked | JPEG PSNR min {min(psnrs):.1f} dB, median {np.median(psnrs):.1f} dB "
          f"| problems: {len(problems)}")
    for p in problems[:50]:
        print("  " + p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
