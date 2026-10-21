#!/usr/bin/env bash
# Crop every photo in <input_dir> to the largest square centered on the face, saved to <output_dir>.
# Intermediate results (detections, centers, frames, manifest, QA sheets, logs) go to work/.
set -euo pipefail
SRC="${1:?usage: run_all.sh <input_dir> <output_dir>}"
OUT="${2:?usage: run_all.sh <input_dir> <output_dir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$HERE/work"
M="$HERE/models"
mkdir -p "$WORK/logs"
py() { GLOG_minloglevel=2 "$HERE/.venv/bin/python" -I "$@"; }  # -I: never import code from the photo folder

echo "== 1. Apple Vision face boxes"
"$HERE/bin/vision_faces" "$SRC" "$WORK/vision.json"
echo "== 2. YuNet boxes + MediaPipe face mesh"
py "$HERE/pipeline/step2_detect.py" "$SRC" "$WORK/vision.json" "$M" "$WORK/detect.json" 2>"$WORK/logs/step2.log"
echo "== 3. Visible face (hair covering a cheek)"
py "$HERE/pipeline/step3_visible_face.py" "$SRC" "$WORK/detect.json" \
   "$M/segmenter/selfie_multiclass_256x256.tflite" "$WORK/visible.json" 2>"$WORK/logs/step3.log"
echo "== 4. Decide each face center"
py "$HERE/pipeline/step4_fuse.py" "$WORK/detect.json" "$WORK/visible.json" "$WORK/overrides.json" "$WORK/centers.json"
echo "== 5. Find frames / bars around photos"
py "$HERE/pipeline/step5_borders.py" "$SRC" "$WORK/borders.json"
echo "== 6. Crop and save"
py "$HERE/pipeline/step6_crop.py" "$SRC" "$WORK/centers.json" "$WORK/borders.json" "$OUT" "$WORK/manifest.csv"
echo "== 7. Verify outputs"
py "$HERE/pipeline/step7_verify.py" "$SRC" "$OUT" "$WORK/manifest.csv"
echo "== QA contact sheets"
py "$HERE/tools/qa_sheets.py" "$OUT" "$WORK/qa"
