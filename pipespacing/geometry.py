"""
Camera + pipe geometry: turn 2D pipe detections and a depth estimate into
3D centerline spacing and edge-to-edge clearance between neighbouring pipes.

Conventions
-----------
* Camera frame: X right, Y down, Z forward (metres), pinhole model.
* Pipes are long cylinders. An axis-aligned box around a pipe that runs
  left-right is wide and short, so its short side is the pipe's apparent
  diameter. Diagonal pipes give near-square boxes and are classified as
  "other" (an oriented-box model would be the fix for those).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Nominal outside diameters (mm) for common NPS steel pipe sizes (ASME B36.10).
NPS_OD_MM = {
    '1/2"': 21.3, '3/4"': 26.7, '1"': 33.4, '1-1/4"': 42.2, '1-1/2"': 48.3,
    '2"': 60.3, '2-1/2"': 73.0, '3"': 88.9, '4"': 114.3, '5"': 141.3,
    '6"': 168.3, '8"': 219.1, '10"': 273.1, '12"': 323.9, '14"': 355.6,
    '16"': 406.4, '18"': 457.0, '20"': 508.0, '24"': 610.0,
}

# Box aspect ratio beyond which a detection is treated as a straight run.
ELONGATION = 2.0

Box = Tuple[float, float, float, float]


# --- Camera -----------------------------------------------------------------

@dataclass
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float = 65.0) -> "Camera":
        """Intrinsics from a horizontal field of view (typical phone ~65-70 deg)."""
        f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
        return cls(f, f, width / 2, height / 2)

    @classmethod
    def from_focal_px(cls, width: int, height: int, focal_px: float) -> "Camera":
        return cls(focal_px, focal_px, width / 2, height / 2)

    @classmethod
    def from_exif(cls, image) -> Optional["Camera"]:
        """Use EXIF FocalLengthIn35mmFilm when the photo carries it."""
        try:
            exif = image.getexif().get_ifd(0x8769)  # Exif sub-IFD
            f35 = float(exif.get(41989) or 0)
        except Exception:
            return None
        if f35 <= 0:
            return None
        w, h = image.size
        # 35mm-equivalent is defined against a 36mm frame width (long side).
        f = f35 / 36.0 * max(w, h)
        return cls(f, f, w / 2, h / 2)

    def backproject(self, u: float, v: float, z: float) -> np.ndarray:
        return np.array([(u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z])


# --- Pipes ------------------------------------------------------------------

@dataclass
class Pipe:
    id: int
    box: Box
    confidence: float
    orientation: str            # "horizontal" | "vertical" | "other"
    diameter_px: float
    depth_m: Optional[float] = None       # camera -> pipe centreline
    od_mm: Optional[float] = None         # outside diameter used / estimated
    nps: Optional[str] = None             # nearest nominal size
    center_3d: Optional[List[float]] = None

    @property
    def center_px(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    def axis_extent(self) -> Tuple[float, float]:
        """Pixel extent along the pipe's run direction."""
        x1, y1, x2, y2 = self.box
        return (y1, y2) if self.orientation == "vertical" else (x1, x2)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["center_px"] = list(self.center_px)
        return d


@dataclass
class PipePair:
    a: int
    b: int
    orientation: str
    spacing_mm: float           # centreline to centreline
    clearance_mm: float         # outer wall to outer wall
    flagged: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def classify(box: Box) -> Tuple[str, float]:
    """Return (orientation, apparent diameter in px) for an axis-aligned box."""
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    if w / h >= ELONGATION:
        return "horizontal", h
    if h / w >= ELONGATION:
        return "vertical", w
    return "other", min(w, h)


def nearest_nps(od_mm: float) -> str:
    return min(NPS_OD_MM, key=lambda k: abs(NPS_OD_MM[k] - od_mm))


def sample_depth(depth: np.ndarray, pipe: Pipe) -> Optional[float]:
    """Median depth over the central core of the pipe's box (avoids edges/background)."""
    x1, y1, x2, y2 = pipe.box
    w, h = x2 - x1, y2 - y1
    # Keep the middle of the run and the middle of the diameter.
    if pipe.orientation == "horizontal":
        fx, fy = 0.25, 0.30
    elif pipe.orientation == "vertical":
        fx, fy = 0.30, 0.25
    else:
        fx, fy = 0.30, 0.30
    H, W = depth.shape
    xa, xb = int(max(0, x1 + w * fx)), int(min(W, x2 - w * fx))
    ya, yb = int(max(0, y1 + h * fy)), int(min(H, y2 - h * fy))
    if xb <= xa or yb <= ya:
        return None
    patch = depth[ya:yb, xa:xb]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(patch)) if patch.size else None


def build_pipes(boxes: Sequence[Box], confidences: Sequence[float]) -> List[Pipe]:
    pipes = []
    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        orientation, d_px = classify(box)
        pipes.append(Pipe(i, tuple(float(v) for v in box), float(conf), orientation, float(d_px)))
    return pipes


def locate_pipes(
    pipes: List[Pipe],
    camera: Camera,
    depth: Optional[np.ndarray] = None,
    od_mm: Optional[float] = None,
) -> str:
    """
    Fill in depth_m / od_mm / center_3d for each pipe. Returns the mode used:

    * "reference": no depth map; all pipes assumed to share the given OD and
      depth comes from similar triangles  Z = f * OD / d_px.
    * "depth":     depth map only; OD is estimated per pipe  OD = d_px * Z / f.
    * "fused":     depth map + known OD. The depth map supplies relative
      structure; its global scale is corrected so the pipes' apparent sizes
      agree with the known OD (monocular metric depth is often off by a
      constant factor).
    """
    if depth is None and not od_mm:
        raise ValueError("need a depth map, a reference OD, or both")

    if depth is None:
        mode = "reference"
        od_m = od_mm / 1000
        for p in pipes:
            p.depth_m = camera.fx * od_m / p.diameter_px
            p.od_mm = od_mm
    else:
        surface = {p.id: sample_depth(depth, p) for p in pipes}
        scale = 1.0
        mode = "depth"
        if od_mm:
            mode = "fused"
            od_m = od_mm / 1000
            ratios = [
                (camera.fx * od_m / p.diameter_px - od_m / 2) / surface[p.id]
                for p in pipes
                if surface[p.id] and p.orientation != "other"
            ]
            if ratios:
                scale = float(np.median(ratios))
        for p in pipes:
            z = surface[p.id]
            if z is None:  # fall back to similar triangles if we have an OD
                if od_mm:
                    p.depth_m = camera.fx * (od_mm / 1000) / p.diameter_px
                    p.od_mm = od_mm
                continue
            z *= scale
            est_od_m = p.diameter_px * z / camera.fx
            # Depth maps see the near surface; the centreline is a radius further.
            p.depth_m = z + (od_mm / 1000 if od_mm else est_od_m) / 2
            p.od_mm = od_mm if od_mm else est_od_m * 1000

    for p in pipes:
        if p.depth_m is None:
            continue
        u, v = p.center_px
        p.center_3d = camera.backproject(u, v, p.depth_m).round(4).tolist()
        if p.od_mm:
            p.od_mm = round(p.od_mm, 1)
            p.nps = nearest_nps(p.od_mm)
        p.depth_m = round(p.depth_m, 3)
    return mode


def _overlap_fraction(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    inter = min(a[1], b[1]) - max(a[0], b[0])
    shorter = min(a[1] - a[0], b[1] - b[0])
    return max(0.0, inter) / shorter if shorter > 0 else 0.0


def neighbour_pairs(pipes: Iterable[Pipe], min_overlap: float = 0.3) -> List[Tuple[Pipe, Pipe]]:
    """
    Adjacent parallel pipes: same orientation, running alongside each other
    (their extents along the run overlap), and next to each other across it.
    """
    pairs = []
    for orientation, across in (("horizontal", 1), ("vertical", 0)):
        group = sorted(
            (p for p in pipes if p.orientation == orientation and p.center_3d),
            key=lambda p: p.center_px[across],
        )
        for i, p in enumerate(group):
            for q in group[i + 1:]:
                if _overlap_fraction(p.axis_extent(), q.axis_extent()) >= min_overlap:
                    pairs.append((p, q))
                    break
    return pairs


def pair_spacing(a: Pipe, b: Pipe, min_clearance_mm: float = 25.0) -> PipePair:
    """Perpendicular distance between the two (parallel) centrelines."""
    axis = np.array([0.0, 1.0, 0.0]) if a.orientation == "vertical" else np.array([1.0, 0.0, 0.0])
    delta = np.array(b.center_3d) - np.array(a.center_3d)
    perp = delta - delta.dot(axis) * axis
    spacing = float(np.linalg.norm(perp)) * 1000
    clearance = spacing - ((a.od_mm or 0) + (b.od_mm or 0)) / 2
    return PipePair(
        a.id, b.id, a.orientation,
        round(spacing, 1), round(clearance, 1),
        flagged=clearance < min_clearance_mm,
    )
