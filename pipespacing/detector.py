"""
Pipe detectors.

* PipeDetector: the fine-tuned single-class YOLOv8 model when it exists,
  falling back to zero-shot YOLO-World before training.
* GroundingDino: open-vocabulary auto-labeler. Much slower than YOLO, but it
  actually knows what a pipe is. YOLO-World scores pipes below 0.05 even on
  photos full of them, because pipes are barely in its training vocabulary.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
FINETUNED = MODELS / "pipe_yolov8n.pt"
WORLD = MODELS / "yolov8s-worldv2.pt"
PROMPTS = ["pipe", "industrial pipe", "metal pipe", "pipeline"]


class PipeDetector:
    def __init__(self, weights: str | Path | None = None, zero_shot: bool = False):
        from ultralytics import YOLO, YOLOWorld

        MODELS.mkdir(exist_ok=True)
        weights = Path(weights) if weights else FINETUNED
        if not zero_shot and weights.exists():
            self.model = YOLO(str(weights))
            self.source = weights.name
        else:
            self.model = YOLOWorld(str(WORLD))
            self.model.set_classes(PROMPTS)
            self.source = f"{WORLD.name} (zero-shot)"

    def __call__(
        self, image: Image.Image, conf: float = 0.25, iou: float = 0.5
    ) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
        # agnostic NMS: every prompt means "pipe", so overlapping hits are one object.
        r = self.model.predict(image, conf=conf, iou=iou, agnostic_nms=True, verbose=False)[0]
        boxes = [tuple(float(v) for v in b) for b in r.boxes.xyxy.tolist()]
        return boxes, [float(c) for c in r.boxes.conf.tolist()]


class GroundingDino:
    model_id = "IDEA-Research/grounding-dino-tiny"
    source = "grounding-dino-tiny (zero-shot)"

    def __init__(self, prompt: str = "pipe.", device: str | None = None):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.prompt = prompt
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device).eval()

    def __call__(
        self, image: Image.Image, conf: float = 0.25, iou: float = 0.5
    ) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        r = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=conf, text_threshold=conf,
            target_sizes=[(image.height, image.width)],
        )[0]
        boxes = [tuple(float(v) for v in b) for b in r["boxes"].tolist()]
        return boxes, [float(s) for s in r["scores"].tolist()]
