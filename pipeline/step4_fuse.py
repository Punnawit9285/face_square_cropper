"""Pick one face center per image and flag images where the detectors disagree.

usage: step4_fuse.py <detect.json> <visible.json> <overrides.json> <out_centers.json>

Face center = center of the bounding box of MediaPipe's FACE_OVAL landmarks. When hair hides part of the face
width (visible width < COVER_LIMIT of the oval), x is shifted to the middle of the visible face (step 3).
Vision and YuNet centers are compared to it after removing each detector's dataset-median offset (their boxes
use different conventions). overrides.json (optional) maps file -> {"cx": .., "cy": .., "note": ..}.
"""
import json
import sys
from pathlib import Path

import numpy as np

RESID_LIMIT = 0.06   # bias-corrected disagreement with Vision / YuNet, in face heights
SPREAD_LIMIT = 0.04  # MediaPipe center spread between the two crop scales, in face heights
IOU_LIMIT = 0.5      # YuNet box must overlap the Vision box this much to count as the same face
COVER_LIMIT = 0.97   # below this visible fraction of the face width, center x on the visible face


def box_center(b):
    return np.array([b["x"] + b["w"] / 2, b["y"] + b["h"] / 2])


def main():
    det = json.loads(Path(sys.argv[1]).read_text())
    visible = json.loads(Path(sys.argv[2]).read_text())
    ov_path = Path(sys.argv[3])
    overrides = json.loads(ov_path.read_text()) if ov_path.exists() else {}
    out_path = Path(sys.argv[4])

    # Each detector draws its box differently (e.g. YuNet includes more forehead); measure that typical offset.
    offsets = {"vision": [], "yunet": []}
    for e in det:
        if "center" in e["mp"]:
            M, h = np.array(e["mp"]["center"]), e["mp"]["height"]
            offsets["vision"].append((box_center(e["vision"]["faces"][0]) - M) / h)
            if e["yunet"]["match"]:
                offsets["yunet"].append((box_center(e["yunet"]["match"]) - M) / h)
    bias = {k: np.median(v, axis=0) for k, v in offsets.items()}

    centers, n_flagged = {}, 0
    for e in det:
        name, m = e["file"], e["mp"]
        flags, resid = [], {}
        estimates = {}
        if e["vision"]["faces"]:
            v = e["vision"]["faces"][0]
            estimates["vision"] = (box_center(v), v["h"])
        if e["yunet"]["match"] and e["yunet"]["match_iou"] >= IOU_LIMIT:
            y = e["yunet"]["match"]
            estimates["yunet"] = (box_center(y), y["h"])
        else:
            flags.append("yunet_no_match")

        if "center" in m:
            M, h = np.array(m["center"]), m["height"]
            source = "mediapipe_oval"
            vf = visible.get(name, {})
            if "vis_shift" in vf and vf["width_cover"] < COVER_LIMIT:
                M = M + [vf["vis_shift"], 0.0]
                source = "mediapipe_oval+visible_shift"
                flags.append("partly_hidden")
            for k, (c, _) in estimates.items():
                resid[k] = float(np.linalg.norm((c - M) / h - bias[k]))
                if resid[k] > RESID_LIMIT:
                    flags.append(f"{k}_disagrees")
            if m["run_spread"] / h > SPREAD_LIMIT:
                flags.append("mediapipe_unstable")
        elif estimates:
            flags.append("mediapipe_missing")
            k = "yunet" if "yunet" in estimates else "vision"
            c, fh = estimates[k]
            h = fh * 0.996 if k == "vision" else fh  # oval height ~= box height; only used for reporting
            M, source = c - bias[k] * h, f"{k}_bias_corrected"
        elif name in overrides:
            flags.append("no_detection")
            M, h, source = None, 0.0, ""  # replaced by the override below
        else:
            raise SystemExit(f"{name}: no face found by any detector; give its center in {ov_path}")

        if e["vision"]["n"] > 1 or e["yunet"]["n"] > 1 or any(r["n_faces"] > 1 for r in m.get("runs", [])):
            flags.append("multiple_faces")

        if name in overrides:
            o = overrides[name]
            M, source = np.array([o["cx"], o["cy"]], dtype=float), "manual_override"

        n_flagged += bool(flags)
        centers[name] = {"w": e["w"], "h": e["h"], "cx": float(M[0]), "cy": float(M[1]), "face_h": float(h),
                         "source": source, "flags": flags, "resid": {k: round(v, 4) for k, v in resid.items()},
                         "corrected": {k: (c - bias[k] * h).round(2).tolist() for k, (c, _) in estimates.items()},
                         "note": overrides.get(name, {}).get("note", "")}

    out_path.write_text(json.dumps(centers, indent=1))
    print(f"fuse: {len(centers)} centers | flagged {n_flagged} | overrides applied "
          f"{sum(c['source'] == 'manual_override' for c in centers.values())} | "
          f"bias vision {bias['vision'].round(3).tolist()} yunet {bias['yunet'].round(3).tolist()} -> {out_path}")


if __name__ == "__main__":
    main()
