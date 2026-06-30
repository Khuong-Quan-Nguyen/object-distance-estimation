# PipeDistance — Object Detection + Monocular Distance Estimation

A small portfolio demo that mirrors the general concept behind an
infrastructure-inspection CV pipeline (object detection → spatial
distance estimation) — rebuilt from scratch using a public pretrained
model and no proprietary data.

> Note: this is a clean-room reimplementation for demo purposes only.
> It does not use, reference, or contain any data, code, or models
> from any employer project under NDA.

## What it does

1. Upload a photo containing two or more objects (bottles, cups,
   laptops, books, etc. — any COCO-trained class).
2. YOLOv8 detects and boxes each object.
3. The app estimates each object's distance from the camera using the
   monocular "similar triangles" method (known real-world width +
   focal length + pixel width in image).
4. It then estimates the real-world distance *between* detected
   objects and draws it on the image.

This is the same conceptual shape as a depth-camera + detection
pipeline, just swapped to a single-camera estimation method so it
works with nothing but a webcam photo.

## Project structure

```
pipedistance/
├── backend/
│   ├── main.py            FastAPI app: detection + distance logic
│   └── requirements.txt
└── frontend/
    └── index.html          Upload UI, calls the backend API
```

## Setup

### 1. Backend

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

First run will auto-download the `yolov8n.pt` pretrained weights
(~6MB) from Ultralytics — no API key needed.

Run the server:

```powershell
uvicorn main:app --reload
```

This starts the API at `http://localhost:8000`.

### 2. Frontend

In a second terminal:

```powershell
cd frontend
python -m http.server 3000
```

Open `http://localhost:3000/index.html` in your browser.

## Calibrating distance for your own camera

The default `FOCAL_LENGTH_PX = 700.0` in `main.py` is a placeholder.
To calibrate it for your actual camera:

1. Place an object of known width (e.g. a 7cm-wide bottle) exactly
   30cm from your camera and take a photo.
2. Run detection on that photo and note the `pixel_width` of the
   bounding box returned in the `detections` list.
3. Compute: `focal_length = (pixel_width * 30) / 7`
4. Enter that value into the "Focal Length" field in the UI (or hard
   code it in `main.py`), and re-run.

You can also add more object classes with known real-world widths to
`KNOWN_WIDTHS_CM` in `main.py` for more accurate readings.

## Next steps (optional, if you want to extend this further)

- Fine-tune YOLOv8 on a small custom dataset of pipe-like objects
  (PVC pipe, cardboard tubes) so it detects "pipe" directly instead
  of relying on generic COCO classes.
- Swap the monocular distance formula for actual stereo or
  depth-camera input if you have access to one (e.g. an Intel
  RealSense), which is closer to the original Kinectrics approach.
- Add a confidence threshold slider and class filter in the UI.

## Talking about this in interviews

This project is a good way to demonstrate the *shape* of the
Kinectrics work (detection → spatial reasoning → actionable distance
output) without touching anything under NDA. Frame it as: "I rebuilt
a simplified, public version of a similar pipeline to demonstrate the
approach."
