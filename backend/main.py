r"""
PipeSpacing API
---------------
    uvicorn backend.main:app --reload        (run from the repo root)

    GET  /                       inspection UI (frontend/index.html)
    GET  /review.html            annotation review UI
    POST /api/analyze            image -> pipes, pair spacing, annotated image
    GET  /api/labels             review queue summary
    GET  /api/labels/{name}      one image's draft / reviewed boxes
    PUT  /api/labels/{name}      save reviewed boxes + status

Environment (all optional, defaults suit local use):
    ALLOWED_ORIGINS   comma-separated origins allowed to call the API from a
                      browser, e.g. https://pipespacing.vercel.app ("*" = any)
    ALLOWED_ORIGIN_REGEX  same, as a pattern, e.g. https://pipespacing(-.*)?\.vercel\.app
                      to also cover Vercel preview deploys
    ENABLE_REVIEW     "0" turns off the label-review endpoints and pages
                      (public deploys: they write to the server's disk)
    MAX_UPLOAD_MB     reject larger uploads (default 15)
    MAX_SIDE          downscale images so the long side is at most this many
                      pixels before inference (default 1600; keeps CPU hosts fast)
"""

from __future__ import annotations

import base64
import io
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipespacing.analyze import analyze  # noqa: E402
from pipespacing.depth import INDOOR, OUTDOOR, DepthAnythingMetric  # noqa: E402
from pipespacing.detector import load_detector  # noqa: E402
from pipespacing.labels import IMAGES, load_labels, save_labels  # noqa: E402

ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
ALLOWED_ORIGIN_REGEX = os.environ.get("ALLOWED_ORIGIN_REGEX") or None
ENABLE_REVIEW = os.environ.get("ENABLE_REVIEW", "1") != "0"
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", 15))
MAX_SIDE = int(os.environ.get("MAX_SIDE", 1600))

app = FastAPI(title="PipeSpacing API")
if ALLOWED_ORIGINS or ALLOWED_ORIGIN_REGEX:
    app.add_middleware(
        CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_origin_regex=ALLOWED_ORIGIN_REGEX,
        allow_methods=["GET", "POST"], allow_headers=["*"],
    )


@lru_cache
def detector():
    return load_detector()


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
    return {"status": "ok", "cuda": torch.cuda.is_available(), "detector": detector().source,
            "review": ENABLE_REVIEW}


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
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"image larger than {MAX_UPLOAD_MB:g} MB")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception:
        raise HTTPException(400, "not an image")
    if mode == "reference" and not od_mm:
        raise HTTPException(400, "reference mode needs od_mm")

    # A focal length given in original pixels must shrink with the image.
    scale = min(1.0, MAX_SIDE / max(image.size))
    if scale < 1:
        image.thumbnail((MAX_SIDE, MAX_SIDE))
        focal_px = focal_px * scale if focal_px else focal_px

    # Inference blocks for seconds on CPU; keep the event loop free meanwhile.
    a = await run_in_threadpool(
        analyze, image, detector(),
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


# --- Annotation review (local only: reads/writes data/ on this machine) ----------

review_api = APIRouter()

class Review(BaseModel):
    status: str
    boxes: List[List[float]]


@review_api.get("/api/labels")
def list_labels():
    labels = load_labels()
    items = [
        {"name": k, "status": v["status"], "n": len(v["boxes"]), "n_auto": len(v["auto"])}
        for k, v in sorted(labels.items())
    ]
    counts = {s: sum(i["status"] == s for i in items) for s in ("pending", "accepted", "rejected")}
    return {"items": items, "counts": counts}


@review_api.get("/api/labels/{name}")
def get_label(name: str):
    labels = load_labels()
    if name not in labels:
        raise HTTPException(404)
    return labels[name]


@review_api.put("/api/labels/{name}")
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


if ENABLE_REVIEW:
    app.include_router(review_api)
    IMAGES.mkdir(parents=True, exist_ok=True)
    app.mount("/images", StaticFiles(directory=IMAGES), name="images")
else:
    @app.get("/review.html", include_in_schema=False)
    def review_disabled():
        raise HTTPException(404)

app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
