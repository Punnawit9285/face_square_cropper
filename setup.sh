#!/usr/bin/env bash
# One-time setup: Python 3.12 env with the detection libraries, the three models, and the Apple Vision tool.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

command -v uv >/dev/null || brew install uv
[ -x "$HERE/.venv/bin/python" ] || uv venv --python 3.12 "$HERE/.venv"
uv pip install --python "$HERE/.venv/bin/python" -r "$HERE/requirements.txt"

fetch() {  # fetch <url> <dest>: download only if missing
  [ -s "$2" ] || { mkdir -p "$(dirname "$2")"; curl -fsSL -o "$2" "$1"; }
}
fetch https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
      "$HERE/models/mediapipe/face_landmarker.task"
fetch https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx \
      "$HERE/models/yunet/face_detection_yunet_2023mar.onnx"
fetch https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite \
      "$HERE/models/segmenter/selfie_multiclass_256x256.tflite"

mkdir -p "$HERE/bin"
swiftc -O "$HERE/pipeline/step1_vision_faces.swift" -o "$HERE/bin/vision_faces"
echo "setup complete"
