"""Crop every image to the largest square whose center is the face center.

usage: step6_crop.py <image_dir> <centers.json> <borders.json> <out_dir> <manifest.csv>

The square must fit inside the photo area: the image minus any frame/bar found by step5_borders.py. Files that are
the same photo in two formats (same name stem and size) share one center and the union of their frame trims,
so they get identical crop boxes.

Pure crop (no resampling, no padding). Each output keeps the source's filename and actual format:
JPEG -> 4:4:4 chroma at the lowest quality in JPEG_QUALITIES that stays >= TARGET_PSNR dB from the cropped
source pixels, same progressive flag, ICC profile / DPI / comment / EXIF kept (EXIF is re-serialized, which
drops the embedded thumbnail of the uncropped photo; pixel-dimension tags are updated).
XMP and Photoshop/IPTC blocks are not carried over. PNG -> lossless, mode (alpha/palette), ICC and DPI kept.
"""
import csv
import io
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image

JPEG_QUALITIES = (95, 97, 99, 100)
TARGET_PSNR = 42.0


def square_box(w, h, cx, cy, trim):
    """Largest square centered on (cx, cy) inside the photo area [left, right) x [top, bottom)."""
    left, top, right, bottom = trim["left"], trim["top"], w - trim["right"], h - trim["bottom"]
    side = math.floor(2 * min(cx - left, right - cx, cy - top, bottom - cy))
    x0 = min(max(round(cx - side / 2), left), right - side)
    y0 = min(max(round(cy - side / 2), top), bottom - side)
    return side, x0, y0


def shared_geometry(centers, borders):
    """Same photo saved in several formats -> one center (mean) and the union of the frame trims."""
    groups = defaultdict(list)
    for name, c in centers.items():
        groups[(Path(name).stem, c["w"], c["h"])].append(name)
    geo = {}
    for members in groups.values():
        cx = sum(centers[m]["cx"] for m in members) / len(members)
        cy = sum(centers[m]["cy"] for m in members) / len(members)
        trim = {s: max(borders[m]["trim"][s] for m in members) for s in ("top", "bottom", "left", "right")}
        for m in members:
            geo[m] = (cx, cy, trim, [o for o in members if o != m])
    return geo


def exif_bytes(src, side):
    exif = src.getexif()
    if not len(exif):
        return None, ""
    orientation = exif.get(ExifTags.Base.Orientation, 1)
    if orientation != 1:
        raise ValueError(f"EXIF orientation {orientation} not handled")
    sub = exif.get_ifd(ExifTags.IFD.Exif)
    for tag in (ExifTags.Base.ExifImageWidth, ExifTags.Base.ExifImageHeight):
        if tag in sub:
            sub[tag] = side
    try:
        return exif.tobytes(), ""
    except Exception as err:  # malformed maker notes etc.: keep the image, drop EXIF
        return None, f"exif dropped ({type(err).__name__})"


def encode_jpeg(out, params):
    ref = np.asarray(out.convert("RGB"), dtype=np.float64)
    for q in JPEG_QUALITIES:
        buf = io.BytesIO()
        out.save(buf, "JPEG", quality=q, subsampling=0, optimize=True, **params)
        dec = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), dtype=np.float64)
        mse = np.mean((dec - ref) ** 2)
        psnr = math.inf if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)
        if psnr >= TARGET_PSNR:
            break
    return buf.getvalue(), q, psnr


def save(src, out, dst, side):
    """Write the crop; returns (note, jpeg_quality, psnr_db)."""
    out.info = {}  # never inherit metadata implicitly; everything kept is passed below
    params = {}
    if src.info.get("icc_profile"):
        params["icc_profile"] = src.info["icc_profile"]
    if src.info.get("dpi"):
        params["dpi"] = src.info["dpi"]
    note = ""
    if src.format == "JPEG":
        exif, note = exif_bytes(src, side)
        if exif:
            params["exif"] = exif
        if src.info.get("comment"):
            params["comment"] = src.info["comment"]
        params["progressive"] = bool(src.info.get("progressive"))
        data, q, psnr = encode_jpeg(out, params)
        dst.write_bytes(data)
        return note, q, round(psnr, 2)
    if src.format == "PNG":
        if "transparency" in src.info:
            params["transparency"] = src.info["transparency"]
        out.save(dst, "PNG", optimize=True, **params)
        return note, "", "lossless"
    raise ValueError(f"unsupported format {src.format}")


def main():
    src_dir, centers_path, borders_path, out_dir, manifest = (Path(a) for a in sys.argv[1:6])
    centers = json.loads(centers_path.read_text())
    borders = json.loads(borders_path.read_text())
    geo = shared_geometry(centers, borders)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, c in centers.items():
        src_path, dst = src_dir / name, out_dir / name
        cx, cy, trim, twins = geo[name]
        with Image.open(src_path) as src:
            src.load()
            w, h = src.size
            assert (w, h) == (c["w"], c["h"]), f"{name}: size changed since detection"
            side, x0, y0 = square_box(w, h, cx, cy, trim)
            if side == w == h:
                shutil.copy2(src_path, dst)
                note, q, psnr = "whole image already centered; copied byte-for-byte", "", "identical"
            else:
                note, q, psnr = save(src, src.crop((x0, y0, x0 + side, y0 + side)), dst, side)
            if twins:
                note = "; ".join(filter(None, [note, f"same photo as {', '.join(twins)}: shared crop box"]))
            rows.append({"file": name, "format": src.format, "mode": src.mode, "src_w": w, "src_h": h,
                         "face_cx": round(cx, 2), "face_cy": round(cy, 2), "side": side, "x0": x0, "y0": y0,
                         "center_err_x": round(x0 + side / 2 - cx, 2),
                         "center_err_y": round(y0 + side / 2 - cy, 2),
                         "frame_trim_tblr": "/".join(str(trim[s]) for s in ("top", "bottom", "left", "right")),
                         "jpeg_quality": q, "psnr_db": psnr,
                         "center_source": c["source"], "flags": ";".join(c["flags"]), "note": note})
    with manifest.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    sides = [r["side"] for r in rows]
    errs = [max(abs(r["center_err_x"]), abs(r["center_err_y"])) for r in rows]
    qs = [r["jpeg_quality"] for r in rows if r["jpeg_quality"] != ""]
    framed = sum(r["frame_trim_tblr"] != "0/0/0/0" for r in rows)
    print(f"crop: {len(rows)} images -> {out_dir} | side {min(sides)}..{max(sides)} px | "
          f"max crop-center error {max(errs):.2f} px | frames excluded in {framed} | "
          f"shared boxes {sum('same photo' in r['note'] for r in rows)} | "
          f"jpeg quality used: { {q: qs.count(q) for q in sorted(set(qs))} }")


if __name__ == "__main__":
    main()
