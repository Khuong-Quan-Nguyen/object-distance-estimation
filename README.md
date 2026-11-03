# PipeSpacing — pipe detection + depth-based spacing for pipe racks

Scan photos of industrial piping, detect each pipe, estimate how far it is
from the camera, and measure the **centreline spacing** and **wall-to-wall
clearance** between neighbouring parallel pipes. Pairs below a minimum
clearance are flagged. Runs one image at a time in a browser or across a
whole folder from the command line.

> Clean-room portfolio rebuild of the *kind* of pipeline I worked on as an
> AI Engineer intern at Kinectrics (detection → depth → real-world distance
> between infrastructure components). Uses only public images (Wikimedia
> Openverse / Wikimedia Commons), public pretrained models, and code written from scratch. It
> contains no data, code, or models from the employer project.

## Pipeline

```
 Openverse / Commons ──► collect_images.py ──► data/raw/images  (+ attribution.csv)
                                                   │
        Grounding DINO "pipe." prompt ──► autolabel.py ──► data/labels.json  (drafts)
                                                   │
                     review.html (accept / fix / reject) ──► reviewed labels
                                                   │
                                  train.py ──► models/pipe_yolov8n.pt
                                                   │
 image ──► detector ──► run direction ──► depth per pipe ──► 3D centrelines ──► spacing / clearance
                              ▲                 ▲
                              │        Depth Anything V2 (metric), or a real
                              │        RGB-D depth frame, or a known pipe OD
                              └── horizontal / vertical / other from box shape
```

| Step | What happens |
|---|---|
| **Collect** | Openverse search over Creative Commons photos (mostly Flickr; no API key), with Wikimedia Commons as an optional top-up. Excludes organs, tobacco pipes, pipe bands, and scanned books. Dedupes by image content and saves author + licence per image. |
| **Auto-label** | Zero-shot Grounding DINO (tiny) with a "pipe." prompt. Drops slivers and whole-scene boxes, and auto-rejects images with no pipe at all (off-topic web photos). YOLO-World was tried first but scores pipes below 0.05 even on photos full of them, so it's kept only as an option. After training, `--labeler finetuned` runs a second labelling round with your own model. |
| **Review** | Browser tool: drag to add a box, click + Del to remove, `a` accept, `r` reject. The original draft is kept, so you can see how much the human pass changed. |
| **Train** | Single-class YOLOv8n fine-tune. Deterministic 80/20 split by file hash. Fits a 4 GB GPU. |
| **Measure** | See *Geometry* below. |

## Geometry

Pinhole camera. Focal length comes from EXIF `FocalLengthIn35mmFilm` when the
photo has it, otherwise from an assumed horizontal FOV (default 65°), or you
can pass `--focal-px` from a calibration.

1. **Run direction.** An axis-aligned box ≥ 2:1 wide is a horizontal run, so
   its height is the pipe's apparent diameter. ≥ 2:1 tall is a vertical run.
   Anything squarer (elbows, diagonals, end-on) is marked `other` and not paired.
2. **Depth per pipe** has three modes:
   * `reference`: every pipe has a known outside diameter (OD), so
     `Z = f · OD / d_px` (similar triangles). No depth model needed.
   * `depth`: median metric depth over the core of the box, plus one radius
     (the depth map sees the near wall). OD is then *estimated* as
     `d_px · Z / f` and snapped to the nearest NPS size.
   * `fused` (default when an OD is given): the depth map supplies the
     relative layout, and its global scale is corrected so pipe sizes agree
     with the known OD. Monocular metric depth is often off by a constant factor.
3. **3D centre:** `X = (u − cx)·Z/f`, `Y = (v − cy)·Z/f`.
4. **Neighbours:** same run direction, sorted across the run, extents along
   the run overlap ≥ 30 %. So pipe 1↔2 and 2↔3 are paired, but not 1↔3.
5. **Spacing** is the component of `P_b − P_a` perpendicular to the run
   axis, which includes depth differences. **Clearance** is
   `spacing − (OD_a + OD_b)/2`.

`tests/test_geometry.py` projects synthetic pipes through a known camera and
checks each mode recovers the true spacing (including a depth map that
reads 20 % short, which `fused` corrects).

## Setup

Everything installs inside the project folder.

```powershell
python -m venv backend\venv
backend\venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130   # or the CPU build
pip install -r requirements.txt
```

First runs download Grounding DINO tiny (~700 MB) and Depth Anything V2
Small (~100 MB) into the Hugging Face cache, plus YOLOv8n weights into `models/`.

## Usage

```powershell
# 1. data
python pipeline\collect_images.py --max 400                     # Openverse (default)
python pipeline\collect_images.py --source commons --max 450    # optional top-up
python pipeline\autolabel.py

# 2. review: start the API, open http://localhost:8010/review.html
uvicorn backend.main:app --port 8010

# 3. train
python pipeline\train.py --epochs 60

# 4. inspect
#    browser:  http://localhost:8010/        (drop one or many images)
#    batch:
python scan.py path\to\photos --od-mm 114.3 --min-clearance-mm 25
python scan.py path\to\photos --depth-dir path\to\depth_frames   # real RGB-D frames (16-bit PNG mm or .npy m)
python scan.py path\to\photos --mode reference --od-mm 168.3     # no depth model
```

`scan.py` writes `runs/scan/annotated/*.jpg`, `pipes.csv`, `pairs.csv`, and `summary.json`.

## Deploying

The static page goes on **Vercel**. The model API (PyTorch, too big for
Vercel's 250 MB function limit) goes on a free **Hugging Face Space** via
the included `Dockerfile`. Step-by-step instructions are in [DEPLOY.md](DEPLOY.md).

## Layout

```
pipespacing/     geometry.py (camera, pipes, spacing) · depth.py · detector.py · analyze.py · labels.py
pipeline/        collect_images.py · autolabel.py · train.py
backend/main.py  FastAPI: /api/analyze, /api/labels, serves frontend/
frontend/        index.html (inspection) · review.html (annotation review)
scan.py          batch CLI
Dockerfile       API image (CPU) for Hugging Face Spaces / Render / Railway
vercel.json      static frontend deploy (scripts/vercel-build.sh writes config.js)
tests/           geometry unit tests
```

## Limitations

* Axis-aligned boxes can't measure diagonal runs. An oriented-box (YOLOv8-OBB)
  or segmentation model would fix this.
* Spacing assumes the camera roughly faces the rack (the run axis is taken as
  the image x or y axis).
* Monocular depth is approximate. Use a known OD (`fused`) or a real depth frame for numbers you'd act on.
* Training labels come from a small reviewed set of web photos, not a curated inspection dataset.

Image credits: see `data/raw/attribution.csv` (generated by the collector).
