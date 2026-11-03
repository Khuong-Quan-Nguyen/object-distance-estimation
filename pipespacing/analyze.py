"""
One image in, pipe spacing out:

    detect pipes -> classify run direction -> depth per pipe -> 3D centreline
    -> spacing / clearance between neighbouring parallel pipes -> annotate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import (
    Camera, Pipe, PipePair, build_pipes, locate_pipes, neighbour_pairs, pair_spacing,
)

OK = (79, 214, 196)
WARN = (232, 80, 60)
DIM = (232, 162, 58)
OTHER = (150, 150, 150)


@dataclass
class Analysis:
    mode: str
    camera: Camera
    pipes: List[Pipe]
    pairs: List[PipePair]
    annotated: Image.Image
    depth: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "focal_px": round(self.camera.fx, 1),
            "pipes": [p.to_dict() for p in self.pipes],
            "pairs": [p.to_dict() for p in self.pairs],
            "flagged": sum(p.flagged for p in self.pairs),
        }


def analyze(
    image: Image.Image,
    detector: Callable,
    depth_fn: Optional[Callable] = None,
    depth_map: Optional[np.ndarray] = None,
    od_mm: Optional[float] = None,
    focal_px: Optional[float] = None,
    hfov_deg: float = 65.0,
    conf: float = 0.25,
    min_clearance_mm: float = 25.0,
) -> Analysis:
    image = image.convert("RGB")
    w, h = image.size
    if focal_px:
        camera = Camera.from_focal_px(w, h, focal_px)
    else:
        camera = Camera.from_exif(image) or Camera.from_fov(w, h, hfov_deg)

    boxes, confs = detector(image, conf=conf)
    pipes = build_pipes(boxes, confs)

    if depth_map is None and depth_fn is not None:
        depth_map = depth_fn(image)
    if depth_map is not None and depth_map.shape != (h, w):
        raise ValueError(f"depth map {depth_map.shape} does not match image {(h, w)}")

    mode = locate_pipes(pipes, camera, depth=depth_map, od_mm=od_mm) if pipes else "none"
    pairs = [pair_spacing(a, b, min_clearance_mm) for a, b in neighbour_pairs(pipes)]
    return Analysis(mode, camera, pipes, pairs, draw(image, pipes, pairs), depth_map)


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _label(draw: ImageDraw.ImageDraw, xy, text, fg, font):
    x, y = xy
    l, t, r, b = draw.textbbox((x, y), text, font=font)
    draw.rectangle((l - 3, t - 2, r + 3, b + 2), fill=(10, 18, 22))
    draw.text((x, y), text, fill=fg, font=font)


def draw(image: Image.Image, pipes: List[Pipe], pairs: List[PipePair]) -> Image.Image:
    out = image.copy()
    d = ImageDraw.Draw(out)
    scale = max(out.size) / 1280
    lw = max(2, round(3 * scale))
    font = _font(max(12, round(16 * scale)))
    by_id = {p.id: p for p in pipes}
    flagged = {i for pr in pairs if pr.flagged for i in (pr.a, pr.b)}

    for p in pipes:
        colour = WARN if p.id in flagged else (OTHER if p.orientation == "other" else OK)
        d.rectangle(p.box, outline=colour, width=lw)
        text = f"P{p.id}"
        if p.nps:
            text += f" {p.nps}"
        if p.depth_m:
            text += f" {p.depth_m:.2f}m"
        _label(d, (p.box[0] + 4, p.box[1] + 4), text, colour, font)

    # Dimension line between each pair, drawn across the shared stretch of run.
    for pr in pairs:
        a, b = by_id[pr.a], by_id[pr.b]
        colour = WARN if pr.flagged else (OTHER if pr.status == "overlap" else DIM)
        (ax, ay), (bx, by) = a.center_px, b.center_px
        if pr.orientation == "horizontal":
            x = (max(a.box[0], b.box[0]) + min(a.box[2], b.box[2])) / 2
            p1, p2 = (x, ay), (x, by)
        else:
            y = (max(a.box[1], b.box[1]) + min(a.box[3], b.box[3])) / 2
            p1, p2 = (ax, y), (bx, y)
        d.line([p1, p2], fill=colour, width=lw)
        for px, py in (p1, p2):
            r = lw * 2
            d.ellipse((px - r, py - r, px + r, py + r), fill=colour)
        mid = ((p1[0] + p2[0]) / 2 + 6, (p1[1] + p2[1]) / 2 - 8)
        gap = "overlap?" if pr.status == "overlap" else f"{pr.clearance_mm:.0f} gap mm"
        _label(d, mid, f"{pr.spacing_mm:.0f} c-c / {gap}", colour, font)
    return out
