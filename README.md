# Face-centered square cropper

Turns every portrait in a folder into a **square whose exact center is the person's face**. Each square is
the largest one that fits, and it is a pure crop: nothing is resized, padded or invented.

Results on `~/Downloads/Uploaded` (509 photos) → `~/Downloads/Uploaded_square`:

| | |
|---|---|
| Faces found | 509 / 509, by all three detectors |
| Centering (faces re-detected on the outputs) | median 0.11 % of the side from dead center, worst 0.92 % (≈1 px on a 119 px image) |
| Crop geometry | crop center within 0.5 px of the face center (pixel rounding) |
| Frames / bars cut away | 49 photos (white mats, black rules, drop shadows, letterbox bars) |
| Hair over a cheek, centered on the visible face | 25 photos |
| Same photo saved as .jpg and .png | 3 pairs, identical crop boxes |
| Verification | 0 problems; JPEG fidelity ≥ 42.2 dB PSNR (median 47.3 dB) |
| Reproducible | `run_all.sh` regenerates all 509 outputs byte-for-byte |

## Run it

```bash
./setup.sh                                   # once: Python env, models, Apple Vision tool
./run_all.sh ~/Downloads/Uploaded ~/Downloads/Uploaded_square
```

To see how one photo is handled, step by step (writes a 4-panel picture like those in `examples/`):

```bash
.venv/bin/python tools/explain_one.py ~/Downloads/Uploaded ~/Downloads/Uploaded_square Alain-Jacquet.jpg examples/me.png
```

## How it works

```
photos ──► 1 Apple Vision boxes ──► 2 YuNet boxes + MediaPipe face mesh ──► 3 visible face ──► 4 face center ─┐
photos ──► 5 frames / bars ──────────────────────────────────────────────────────────────────────────────────┤
                                                                  6 largest centered square, crop, save ◄─────┘
                                                                  7 verify every output
```

Each step is a small program that reads the previous step's JSON in `work/` and writes its own, so every
intermediate result can be opened and inspected.

### Step 1: Apple Vision face boxes (`pipeline/step1_vision_faces.swift`)
macOS's built-in face detector (`VNDetectFaceRectanglesRequest`) returns a box for every face, largest
first. It is fast, needs no download, and seeds the next step: the largest face is the subject (one photo
has a guest in the background, and this rule correctly ignores them).

### Step 2: YuNet + MediaPipe face mesh (`pipeline/step2_detect.py`)
- **YuNet** (`yunet_faces`): an independent OpenCV detector. Its box is used only as a cross-check.
- **MediaPipe Face Landmarker** (`mp_oval`): fits a 478-point 3D mesh to the face. It runs on a square
  crop around the subject, upscaled to 512 px, so even a tiny face in a big photo fills the model's input. It runs
  twice (crop 2× and 3× the face size) and averages the two; they agree within 1 % of the face height
  on 99 % of photos.
- The 36 **FACE_OVAL** points trace the face outline: forehead top, cheeks, jaw, chin.

### Step 3: Visible face (`pipeline/step3_visible_face.py`)
When hair falls over a cheek, the mesh still draws the hidden cheek under the hair, which would put the
"center" off to one side of what you actually see. `measure()` runs MediaPipe's skin segmenter. Then, for
each pixel row from the eyes to the mouth, it compares the middle of the visible face skin inside the outline
with the middle of the outline. The median difference is `vis_shift`. It is 0 when nothing covers the face.

### Step 4: Decide the face center (`pipeline/step4_fuse.py`)
- **Face center = middle of the face outline's bounding box** (forehead↔chin, cheek↔cheek). That point
  sits on the nose line, about mid-nose.
- If more than 3 % of the face width is hidden (`COVER_LIMIT`), x moves by `vis_shift` to the middle of
  the visible face. Vertical position never changes.
- Cross-check: Vision and YuNet draw boxes differently (YuNet includes more forehead), so their typical
  offset is measured over all photos and removed. Anything still more than 6 % of the face height away
  is **flagged** for a human look (`tools/review_sheets.py`). Every flag in this batch was reviewed. The
  flags were caused by box conventions (Vision drifts toward the nose on turned heads), not by a wrong center.
- `work/overrides.json` (optional) lets you pin a center by hand: `{"file.jpg": {"cx": 120, "cy": 95}}`.

### Step 5: Frames and bars (`pipeline/step5_borders.py`)
Some originals are framed thumbnails (white mat + black rule + shadow) or have black bars. For each side,
`border_width()` walks inward line by line. A frame is a run of **flat, colorless** lines (white/black/gray)
ending at a **straight edge across at least half of the line**. Colored design elements, like a navy
poster band, don't qualify. `trim` adds 1 px so JPEG blur along the frame line can't leave a faint stripe.

### Step 6: Largest centered square (`pipeline/step6_crop.py`)
- `square_box()`: inside the photo area (image minus frames), the side is **twice the shortest distance
  from the face center to an edge**. That is the biggest square that keeps the face dead-center.
- `shared_geometry()`: files with the same name and size (the same photo as .jpg and .png) share one
  center, so their crops are identical.
- Saving keeps each file's real format and name:
  - **JPEG:** 4:4:4 color at quality 95, stepping up to 97 / 99 / 100 until the crop is ≥ 42 dB from the
    original pixels (`encode_jpeg`).
  - **PNG:** lossless; alpha and palette are kept.
  - **ICC color profile:** kept (Display P3 / Adobe RGB files keep their colors).
  - **EXIF:** kept, minus the embedded thumbnail of the *uncropped* photo. Pixel-size tags are updated.
  - **Dropped:** XMP and Photoshop blocks, which carry old sizes and previews.
- `work/manifest.csv` records every decision: center, box, frame trim, JPEG quality, PSNR, flags.

### Step 7: Verify (`pipeline/step7_verify.py`)
For every output, it checks the following:
- Square, with the planned size.
- Same format, mode and ICC profile as the original.
- No stale thumbnail, XMP or IPTC.
- Pixels match the original at exactly the planned box: PNG bit-exact, JPEG ≥ 40 dB.
- Every box shifted by 1 px matches *worse*, which proves there is no off-by-one.

It exits non-zero if anything fails.

### Tools
- `tools/explain_one.py`: the 4-panel explanation of one photo (see `examples/`).
- `tools/qa_sheets.py`: contact sheets of all outputs with a crosshair through each center
  (`work/qa/`). Every sheet was reviewed by eye.
- `tools/review_sheets.py`: for flagged photos, shows the face outline and every detector's center side by side.

## Code walkthrough

The code is seven small programs that `run_all.sh` runs in order. Each one reads the previous step's JSON
from `work/` and writes its own, so you can stop after any step and open its output. All coordinates are
pixels of the original photo, measured from its top-left corner.

| Step | File | Reads | Writes | Start reading at |
|---|---|---|---|---|
| 1 | `pipeline/step1_vision_faces.swift` | photos | `vision.json` | the loop over files |
| 2 | `pipeline/step2_detect.py` | photos, `vision.json` | `detect.json` | `mp_oval()`, `yunet_faces()` |
| 3 | `pipeline/step3_visible_face.py` | photos, `detect.json` | `visible.json` | `measure()` |
| 4 | `pipeline/step4_fuse.py` | `detect.json`, `visible.json`, `overrides.json` | `centers.json` | `main()` |
| 5 | `pipeline/step5_borders.py` | photos | `borders.json` | `border_width()` |
| 6 | `pipeline/step6_crop.py` | photos, `centers.json`, `borders.json` | cropped photos, `manifest.csv` | `square_box()` |
| 7 | `pipeline/step7_verify.py` | photos, cropped photos, `manifest.csv` | exit code | the shifted-box loop |

Following one photo through the code:

1. **Find faces fast.** Apple Vision returns a box for every face, largest first, and the largest is the
   subject. Vision measures from the bottom-left corner, so each box is flipped to top-left coordinates.
2. **Get a precise outline.** `yunet_faces()` runs a second detector and keeps the box that overlaps
   Vision's most (a cross-check only). `mp_oval()` crops a square around the face, enlarges it to 512 px
   and fits the 478-point mesh. The center of the box around the 36 outline points is the face center.
   It runs at 2× and 3× crop sizes, and `main()` averages the two.
3. **Correct for hair.** `measure()` keeps the segmenter's face-skin pixels that fall inside the outline.
   For each row between the eyes and the mouth, it compares the middle of the visible skin with the middle
   of the outline. The median difference is `vis_shift`.
4. **Pick the center.** The center is MediaPipe's, moved sideways by `vis_shift` when less than
   `COVER_LIMIT` of the face width is visible. The first loop in `main()` measures each cross-check
   detector's typical offset over all photos. The second loop subtracts it and flags anything still more
   than `RESID_LIMIT` away. If MediaPipe found nothing, the center falls back to YuNet, then Vision. A
   center in `work/overrides.json` replaces all of these, and a photo with no face and no override stops
   the run.
5. **Find frames.** `border_width()` walks inward from each edge, line by line, and stops at the first line
   that isn't flat. The frame ends at the innermost sharp edge in that flat run, as long as the run is
   white, black or gray.
6. **Crop.** `square_box()` sets the side to twice the shortest distance from the center to the edge of
   the frame-free area: the largest square that keeps the face centered. `src.crop()` cuts it without
   resizing. `encode_jpeg()` raises JPEG quality until the crop is within `TARGET_PSNR` of the original,
   and PNG is lossless. If the square is the whole image, the file is copied unchanged.
7. **Verify.** Each output's size, format, mode, ICC profile, DPI and metadata are checked, and its pixels
   are compared with the source at the planned box. Then the box is shifted 1 px in all 8 directions, and
   every shifted box must match worse, which rules out off-by-one errors.

**Limitation:** a photo with an EXIF rotation tag (for example, a phone photo shot sideways) stops the
run with an error instead of being rotated (`load_rgb()` in step 2, `exif_bytes()` in step 6). Rotate
such photos before running.

## Tuning knobs

| Constant | File | Meaning |
|---|---|---|
| `COVER_LIMIT = 0.97` | step4 | apply the visible-face shift when < 97 % of the face width is visible |
| `RESID_LIMIT = 0.06` | step4 | flag when a cross-check detector is > 6 % of face height away |
| `MP_EXPANDS = (2.0, 3.0)` | step2 | MediaPipe crop sizes, in face sizes |
| `MAX_DEPTH`, `FLAT_STD`, `CHROMA_MAX`, `EDGE_MIN` | step5 | how deep/flat/gray a frame can be, and how sharp its edge |
| `JPEG_QUALITIES`, `TARGET_PSNR` | step6 | JPEG quality ladder and fidelity target |

## Files

```
setup.sh, run_all.sh, requirements.txt
pipeline/   step1 … step7 (the cropper itself)
tools/      explain_one.py, qa_sheets.py, review_sheets.py
models/     face_landmarker.task, face_detection_yunet_2023mar.onnx, selfie_multiclass_256x256.tflite
            (downloaded by setup.sh; not in git)
bin/        vision_faces, compiled from step 1 by setup.sh (not in git)
work/       outputs of the last run: *.json, manifest.csv, qa/ contact sheets, logs/
            (written by run_all.sh; not in git)
examples/   step-by-step explanations of five representative photos
```
