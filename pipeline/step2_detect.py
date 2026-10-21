"""Locate the subject face in every image with OpenCV YuNet and the MediaPipe Face Landmarker.

usage: step2_detect.py <image_dir> <vision.json> <models_dir> <out.json>

The subject is seeded by the largest Apple Vision face. MediaPipe runs on face-centred crops at two
scales so even small/far faces fill the model input; its 36 FACE_OVAL points give the face outline.
All coordinates are source pixels on the raw pixel grid (top-left origin).
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


def _oval_indices():
    idx = []
    for c in mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL:
        for v in (c.start, c.end):
            if v not in idx:
                idx.append(v)
    return idx


FACE_OVAL = _oval_indices()
KEYPOINTS = {"forehead": 10, "chin": 152, "nose_tip": 1, "left_iris": 468, "right_iris": 473,
             "mouth_left": 61, "mouth_right": 291}
MP_EXPANDS = (2.0, 3.0)  # crop side as a multiple of the seed face size
MP_SIDE = 512


def load_rgb(path):
    """Decode to an RGB array; transparent pixels are composited on white (detection only)."""
    im = Image.open(path)
    im.load()
    orientation = im.getexif().get(0x0112, 1)
    if orientation not in (0, 1):
        raise ValueError(f"EXIF orientation {orientation} not handled")
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    return np.ascontiguousarray(np.asarray(im.convert("RGB")))


def yunet_faces(detector, rgb):
    h, w = rgb.shape[:2]
    long_side = max(h, w)
    s = 1600 / long_side if long_side > 1600 else (640 / long_side if long_side < 640 else 1.0)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if s != 1.0:
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
        bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=interp)
    sx, sy = bgr.shape[1] / w, bgr.shape[0] / h
    detector.setInputSize((bgr.shape[1], bgr.shape[0]))
    _, faces = detector.detect(bgr)
    out = []
    for f in faces if faces is not None else []:
        lm = f[4:14].reshape(5, 2) / [sx, sy]
        out.append({"x": float(f[0] / sx), "y": float(f[1] / sy), "w": float(f[2] / sx), "h": float(f[3] / sy),
                    "lm5": lm.round(2).tolist(), "score": float(f[14])})
    out.sort(key=lambda d: d["w"] * d["h"], reverse=True)
    return out


def iou(a, b):
    ax1, ay1, bx1, by1 = a["x"] + a["w"], a["y"] + a["h"], b["x"] + b["w"], b["y"] + b["h"]
    iw = max(0.0, min(ax1, bx1) - max(a["x"], b["x"]))
    ih = max(0.0, min(ay1, by1) - max(a["y"], b["y"]))
    inter = iw * ih
    return inter / (a["w"] * a["h"] + b["w"] * b["h"] - inter)


def mp_oval(landmarker, rgb, seed, expand):
    """Run the landmarker on a square crop centred on the seed face; return the face nearest the crop centre."""
    H, W = rgb.shape[:2]
    cx, cy = seed["x"] + seed["w"] / 2, seed["y"] + seed["h"] / 2
    side = int(round(expand * max(seed["w"], seed["h"])))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    sub = rgb[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    pads = (max(0, -y0), max(0, y1 - H), max(0, -x0), max(0, x1 - W))
    if any(pads):
        sub = cv2.copyMakeBorder(sub, *pads, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    interp = cv2.INTER_AREA if side > MP_SIDE else cv2.INTER_CUBIC
    sub = cv2.resize(sub, (MP_SIDE, MP_SIDE), interpolation=interp)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(sub)))
    best = None
    for pts in res.face_landmarks:
        xy = np.array([[p.x, p.y] for p in pts]) * side + [x0, y0]
        ov = xy[FACE_OVAL]
        box = [ov[:, 0].min(), ov[:, 1].min(), ov[:, 0].max(), ov[:, 1].max()]
        c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        d = float(np.hypot(c[0] - cx, c[1] - cy))
        if best is None or d < best["dist_to_seed"]:
            best = {"expand": expand, "n_faces": len(res.face_landmarks), "dist_to_seed": d,
                    "center": [float(c[0]), float(c[1])], "box": [float(v) for v in box], "xy": xy}
    return best


def main():
    image_dir, vision_json, models_dir, out_json = (Path(a) for a in sys.argv[1:5])
    vision = {e["file"]: e for e in json.loads(vision_json.read_text())}
    yunet = cv2.FaceDetectorYN.create(str(models_dir / "yunet/face_detection_yunet_2023mar.onnx"), "", (640, 640),
                                      0.6, 0.3, 50)
    landmarker = mp_vision.FaceLandmarker.create_from_options(mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(models_dir / "mediapipe/face_landmarker.task")),
        running_mode=mp_vision.RunningMode.IMAGE, num_faces=3,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3))

    files = sorted(p.name for p in image_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    results = []
    for i, name in enumerate(files, 1):
        rgb = load_rgb(image_dir / name)
        H, W = rgb.shape[:2]
        v = vision.get(name)
        assert v and (v["w"], v["h"]) == (W, H), f"{name}: vision size mismatch"
        entry = {"file": name, "w": W, "h": H, "vision": {"n": len(v["faces"]), "faces": v["faces"]}}

        yfaces = yunet_faces(yunet, rgb)
        seed = v["faces"][0] if v["faces"] else (yfaces[0] if yfaces else None)
        match = max(yfaces, key=lambda f: iou(f, seed), default=None) if seed else None
        entry["yunet"] = {"n": len(yfaces), "faces": yfaces[:5],
                          "match": match, "match_iou": iou(match, seed) if match else 0.0}

        runs = [r for r in (mp_oval(landmarker, rgb, seed, e) for e in MP_EXPANDS) if r] if seed else []
        mpd = {"runs": [{k: r[k] for k in ("expand", "n_faces", "dist_to_seed", "center", "box")} for r in runs]}
        if runs:
            xy = np.mean([r["xy"] for r in runs], axis=0)
            box = np.mean([r["box"] for r in runs], axis=0)
            mpd.update({
                "center": [float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)],
                "box": box.round(2).tolist(),
                "width": float(box[2] - box[0]), "height": float(box[3] - box[1]),
                "run_spread": float(np.hypot(*np.subtract(runs[0]["center"], runs[-1]["center"]))),
                "keypoints": {k: xy[j].round(2).tolist() for k, j in KEYPOINTS.items()},
                "oval": xy[FACE_OVAL].round(2).tolist(),
            })
        entry["mp"] = mpd
        results.append(entry)
        if i % 50 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    out_json.write_text(json.dumps(results, indent=1))
    n_mp = sum(1 for r in results if "center" in r["mp"])
    n_y = sum(1 for r in results if r["yunet"]["match"] and r["yunet"]["match_iou"] > 0.3)
    print(f"detect: {len(results)} images | mediapipe ok {n_mp} | yunet matched {n_y} -> {out_json}")


if __name__ == "__main__":
    main()
