"""
PipeSpacing API
---------------
    uvicorn backend.main:app --reload        (run from the repo root)

    GET  /                       inspection UI (frontend/index.html)
    GET  /review.html            annotation review UI
    POST /api/analyze            image -> pipes, pair spacing, annotated image
    GET  /api/labels             review queue summary
    GET  /api/labels/{name}      one image's draft / reviewed boxes
    PUT  /api/labels/{name}      save reviewed boxes + status
"""

from __future__ import annotations

import base64
import io
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipespacing.analyze import analyze  # noqa: E402
from pipespacing.depth import INDOOR, OUTDOOR, DepthAnythingMetric  # noqa: E402
from pipespacing.detector import PipeDetector  # noqa: E402
from pipespacing.labels import IMAGES, load_labels, save_labels  # noqa: E402

app = FastAPI(title="PipeSpacing API")


@lru_cache
def detector() -> PipeDetector:
    return PipeDetector()


@lru_cache
def depth_model(outdoor: bool) -> DepthAnythingMetric:
    return DepthAnythingMetric(OUTDOOR if outdoor else INDOOR)


def b64_jpeg(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def depth_preview(depth: np.ndarray) -> Image.Image:
    """Near = bright, far = dark, robust to outliers."""
    lo, hi = np.nanpercentile(depth, [2, 98])
    norm = 1 - np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.stack([norm * 232, norm * 162 + (1 - norm) * 28, norm * 58 + (1 - norm) * 34], -1)
    return Image.fromarray(np.nan_to_num(rgb).astype(np.uint8))


@app.get("/api/health")
def health():
    import torch
    return {"status": "ok", "cuda": torch.cuda.is_available(), "detector": detector().source}


@app.post("/api/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    mode: str = Form("auto"),                 # auto | depth | reference
    od_mm: Optional[float] = Form(None),
    focal_px: Optional[float] = Form(None),
    hfov_deg: float = Form(65.0),
    min_clearance_mm: float = Form(25.0),
    conf: float = Form(0.25),
    outdoor: bool = Form(False),
):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "not an image")
    if mode == "reference" and not od_mm:
        raise HTTPException(400, "reference mode needs od_mm")

    a = analyze(
        image, detector(),
        depth_fn=None if mode == "reference" else depth_model(outdoor),
        od_mm=od_mm or None, focal_px=focal_px or None, hfov_deg=hfov_deg,
        conf=conf, min_clearance_mm=min_clearance_mm,
    )
    out = a.to_dict()
    out["detector"] = detector().source
    out["annotated_image_base64"] = b64_jpeg(a.annotated)
    if a.depth is not None:
        out["depth_image_base64"] = b64_jpeg(depth_preview(a.depth))
    return out


# --- Annotation review --------------------------------------------------------

class Review(BaseModel):
    status: str
    boxes: List[List[float]]


@app.get("/api/labels")
def list_labels():
    labels = load_labels()
    items = [
        {"name": k, "status": v["status"], "n": len(v["boxes"]), "n_auto": len(v["auto"])}
        for k, v in sorted(labels.items())
    ]
    counts = {s: sum(i["status"] == s for i in items) for s in ("pending", "accepted", "rejected")}
    return {"items": items, "counts": counts}


@app.get("/api/labels/{name}")
def get_label(name: str):
    labels = load_labels()
    if name not in labels:
        raise HTTPException(404)
    return labels[name]


@app.put("/api/labels/{name}")
def put_label(name: str, review: Review):
    if review.status not in ("pending", "accepted", "rejected"):
        raise HTTPException(400, "bad status")
    labels = load_labels()
    if name not in labels:
        raise HTTPException(404)
    e = labels[name]
    W, H = e["width"], e["height"]
    e["boxes"] = [
        [round(max(0, min(W, x1)), 1), round(max(0, min(H, y1)), 1),
         round(max(0, min(W, x2)), 1), round(max(0, min(H, y2)), 1)]
        for x1, y1, x2, y2 in (sorted_box(b) for b in review.boxes)
        if abs(x2 - x1) >= 3 and abs(y2 - y1) >= 3
    ]
    e["status"] = review.status
    e["auto_rejected"] = False  # a human has looked at it now
    save_labels(labels)
    return {"ok": True, "n": len(e["boxes"])}


def sorted_box(b):
    x1, y1, x2, y2 = b[:4]
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


IMAGES.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=IMAGES), name="images")
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
