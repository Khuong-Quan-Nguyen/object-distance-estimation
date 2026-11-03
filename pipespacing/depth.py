"""
Depth sources. Anything that maps an RGB image to an (H, W) array of metres.

* DepthAnythingMetric: monocular metric depth (Depth Anything V2), standing in
  for a depth camera when all you have is a photo.
* load_depth_file:     a real aligned depth frame from an RGB-D camera, e.g.
  a RealSense 16-bit PNG in millimetres, or a .npy array in metres.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

INDOOR = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
OUTDOOR = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


class DepthAnythingMetric:
    def __init__(self, model_id: str = INDOOR, device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device).eval()

    def __call__(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        depth = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)]
        )[0]["predicted_depth"]
        return depth.float().cpu().numpy()


def load_depth_file(path: str | Path, png_scale: float = 0.001) -> np.ndarray:
    """Load an aligned depth frame. 16-bit PNGs are scaled by png_scale (mm -> m)."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    raw = np.asarray(Image.open(path)).astype(np.float32)
    raw[raw == 0] = np.nan  # 0 = no return on most depth sensors
    return raw * png_scale
