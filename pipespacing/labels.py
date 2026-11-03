"""
Label store for the semi-automated annotation loop.

data/labels.json maps image file -> {
    status: "pending" | "accepted" | "rejected",
    width, height,
    boxes: [[x1, y1, x2, y2], ...]          # current (human-edited) boxes
    auto:  [[x1, y1, x2, y2, conf], ...]    # original auto-label draft
}
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
IMAGES = DATA / "raw" / "images"
LABELS = DATA / "labels.json"


def load_labels() -> dict:
    if LABELS.exists():
        return json.loads(LABELS.read_text(encoding="utf-8"))
    return {}


def save_labels(labels: dict) -> None:
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    tmp = LABELS.with_suffix(".tmp")
    tmp.write_text(json.dumps(labels, indent=1), encoding="utf-8")
    os.replace(tmp, LABELS)  # atomic, so a crash never leaves half a file
