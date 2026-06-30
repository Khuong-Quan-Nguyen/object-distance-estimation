"""
PipeDistance — Object Detection + Monocular Distance Estimation Demo
----------------------------------------------------------------------
A lightweight, public-data version of an inspection-style CV pipeline:
detect objects in an image, then estimate the real-world distance
between them using a single RGB camera (no depth sensor required).

Pipeline:
  1. YOLOv8 (pretrained, COCO classes) detects objects in the uploaded image.
  2. For each detected object, we estimate its distance FROM THE CAMERA
     using the classic "similar triangles" monocular distance formula:

         distance = (known_real_width * focal_length) / pixel_width

  3. Once each object has a camera distance + pixel center, we
     triangulate the real-world distance BETWEEN any two objects.
  4. The annotated image (boxes + a connecting line + distance label)
     is returned to the frontend.

This is a toy/demo project — not the proprietary Kinectrics pipeline,
just the same general concept (detection -> spatial reasoning) rebuilt
from scratch on public models and your own photos.
"""

import io
import math
from typing import List, Tuple

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

app = FastAPI(title="PipeDistance API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pretrained YOLOv8 nano model — downloads automatically on first run.
# Swap this for a fine-tuned .pt file later if you train on your own
# "pipe-like" objects (PVC pipes, bottles, tubes, etc.)
model = YOLO("yolov8n.pt")

# --- Calibration constants -------------------------------------------
# To get real distances you need to calibrate your camera once:
#   1. Place an object of KNOWN_WIDTH_CM at a KNOWN_DISTANCE_CM from the camera.
#   2. Measure how wide it appears in pixels in the photo (pixel_width).
#   3. FOCAL_LENGTH = (pixel_width * KNOWN_DISTANCE_CM) / KNOWN_WIDTH_CM
# Defaults below are rough placeholders — recalibrate for your own camera.
FOCAL_LENGTH_PX = 700.0

# Known real-world widths (cm) for common COCO classes, used as the
# reference object size in the distance formula. Add more as needed.
KNOWN_WIDTHS_CM = {
    "bottle": 7.0,
    "cup": 8.0,
    "cell phone": 7.5,
    "book": 15.0,
    "laptop": 30.0,
    "remote": 5.0,
    "mouse": 6.0,
}
DEFAULT_WIDTH_CM = 10.0  # fallback for classes not in the table above


def estimate_distance_from_camera(pixel_width: float, class_name: str) -> float:
    real_width = KNOWN_WIDTHS_CM.get(class_name, DEFAULT_WIDTH_CM)
    if pixel_width <= 0:
        return 0.0
    return (real_width * FOCAL_LENGTH_PX) / pixel_width


def box_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect")
async def detect(file: UploadFile = File(...), focal_length: float = Form(None)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    focal = focal_length if focal_length else FOCAL_LENGTH_PX

    results = model(image)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        pixel_width = x2 - x1
        cam_distance = estimate_distance_from_camera(pixel_width, class_name)
        cx, cy = box_center((x1, y1, x2, y2))

        detections.append({
            "class_name": class_name,
            "confidence": round(conf, 3),
            "box": [x1, y1, x2, y2],
            "center": [cx, cy],
            "pixel_width": pixel_width,
            "distance_from_camera_cm": round(cam_distance, 1),
        })

    # Compute pairwise real-world distance between detected objects
    pairwise = []
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            a, b = detections[i], detections[j]
            # Triangulate using camera distance + pixel offset (simplified
            # geometric approximation, good enough for a demo).
            d1, d2 = a["distance_from_camera_cm"], b["distance_from_camera_cm"]
            avg_dist = (d1 + d2) / 2
            pixel_gap = math.dist(a["center"], b["center"])
            # Convert pixel gap to real-world cm using the average object's
            # known-width-to-pixel-width ratio as a local scale factor.
            scale = KNOWN_WIDTHS_CM.get(a["class_name"], DEFAULT_WIDTH_CM) / max(a["pixel_width"], 1)
            real_gap_cm = pixel_gap * scale
            pairwise.append({
                "object_a": a["class_name"],
                "object_b": b["class_name"],
                "estimated_distance_cm": round(real_gap_cm, 1),
            })

    # Draw annotations
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        label = f'{d["class_name"]} {d["confidence"]:.2f} | {d["distance_from_camera_cm"]}cm'
        draw.text((x1, max(0, y1 - 14)), label, fill="lime")

    centers = [d["center"] for d in detections]
    pair_idx = 0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            draw.line([centers[i], centers[j]], fill="yellow", width=2)
            mid = ((centers[i][0] + centers[j][0]) / 2, (centers[i][1] + centers[j][1]) / 2)
            dist_label = pairwise[pair_idx]["estimated_distance_cm"]
            draw.text(mid, f"{dist_label}cm", fill="yellow")
            pair_idx += 1

    buf = io.BytesIO()
    annotated.save(buf, format="JPEG")
    annotated_bytes = buf.getvalue()

    import base64
    encoded_image = base64.b64encode(annotated_bytes).decode("utf-8")

    return JSONResponse({
        "detections": detections,
        "pairwise_distances": pairwise,
        "annotated_image_base64": encoded_image,
    })
