"""Measure the horizontal center of the *visible* face (face-skin pixels inside the MediaPipe face oval).

usage: step3_visible_face.py <image_dir> <detect.json> <segmenter.tflite> <out.json>

For each pixel row between the eye line and the mouth line, compare the midpoint of the visible face-skin
pixels inside the oval with the midpoint of the oval itself. The median difference (vis_shift, source px) is
how far the visible face sits off-center within the face outline: 0 when nothing covers the face, and half
the hidden width when hair covers one cheek. width_cover is the median visible fraction of the oval width.
"""
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image

SIDE = 512
FACE_SKIN = 3


def load_rgb(path):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    return np.ascontiguousarray(np.asarray(im.convert("RGB")))


def create_segmenter(model):
    return mp_vision.ImageSegmenter.create_from_options(mp_vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)), running_mode=mp_vision.RunningMode.IMAGE,
        output_category_mask=True, output_confidence_masks=False))


def measure(seg, rgb, m, debug=False):
    """Visible-face measurement for one image. m is the image's "mp" entry from detect.json.

    With debug=True the result also holds the masks (in a SIDE x SIDE face crop) and the crop geometry.
    """
    H, W = rgb.shape[:2]
    cx, cy = m["center"]
    side = int(round(2.0 * max(m["width"], m["height"])))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    sub = rgb[max(0, y0):min(H, y0 + side), max(0, x0):min(W, x0 + side)]
    pads = (max(0, -y0), max(0, y0 + side - H), max(0, -x0), max(0, x0 + side - W))
    if any(pads):
        sub = cv2.copyMakeBorder(sub, *pads, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    sub = cv2.resize(sub, (SIDE, SIDE), interpolation=cv2.INTER_AREA if side > SIDE else cv2.INTER_CUBIC)
    cat = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(sub))).category_mask
    skin = cat.numpy_view().squeeze() == FACE_SKIN

    k = SIDE / side
    to_crop = lambda p: (np.asarray(p, dtype=float) - [x0, y0]) * k
    oval = np.zeros((SIDE, SIDE), np.uint8)
    cv2.fillPoly(oval, [np.round(to_crop(m["oval"])).astype(np.int32)], 1)
    visible = skin & oval.astype(bool)
    visible = cv2.morphologyEx(visible.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)

    kp = m["keypoints"]
    eye_y = to_crop([0, (kp["left_iris"][1] + kp["right_iris"][1]) / 2])[1]
    mouth_y = to_crop([0, (kp["mouth_left"][1] + kp["mouth_right"][1]) / 2])[1]
    shifts, cover = [], []
    for r in range(int(round(eye_y)), int(round(mouth_y)) + 1):
        ov = np.flatnonzero(oval[r])
        xs = np.flatnonzero(visible[r])
        if len(ov) == 0 or len(xs) < 0.3 * len(ov):
            continue
        shifts.append((xs[0] + xs[-1]) / 2 - (ov[0] + ov[-1]) / 2)
        cover.append((xs[-1] - xs[0] + 1) / (ov[-1] - ov[0] + 1))
    res = {"skin_in_oval": float(visible.sum() / max(1, oval.sum())), "rows": len(shifts)}
    if shifts:
        res["vis_shift"] = float(np.median(shifts) / k)
        res["width_cover"] = float(np.median(cover))
    if debug:
        res["debug"] = {"x0": x0, "y0": y0, "side": side, "visible": visible, "oval": oval.astype(bool),
                        "rows": (int(round(eye_y)), int(round(mouth_y)))}
    return res


def main():
    image_dir, det_path, model, out_path = (Path(a) for a in sys.argv[1:5])
    det = json.loads(det_path.read_text())
    seg = create_segmenter(model)
    out = {e["file"]: measure(seg, load_rgb(image_dir / e["file"]), e["mp"]) for e in det}
    out_path.write_text(json.dumps(out, indent=1))
    ok = sum("vis_shift" in r for r in out.values())
    print(f"visible_face: {len(out)} images, measured {ok} -> {out_path}")


if __name__ == "__main__":
    main()
