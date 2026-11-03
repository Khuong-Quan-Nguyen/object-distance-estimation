# PipeSpacing API, CPU-only. Works as a Hugging Face Docker Space (port 7860,
# uid 1000) and on Render / Railway / Fly (they set $PORT).
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/hf \
    YOLO_CONFIG_DIR=/home/user/.ultralytics \
    YOLO_AUTOINSTALL=false \
    PYTHONUNBUFFERED=1 \
    ENABLE_REVIEW=0 \
    PORT=7860
WORKDIR /home/user/app

# CPU wheels keep the image ~1.5 GB instead of ~6 GB with CUDA.
COPY --chown=user requirements-deploy.txt .
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements-deploy.txt

COPY --chown=user pipespacing pipespacing
COPY --chown=user backend/main.py backend/main.py
COPY --chown=user frontend frontend
COPY --chown=user models models

# Bake weights into the image so a cold start doesn't re-download ~800 MB.
# Grounding DINO is only used until models/pipe_yolov8n.pt is committed.
RUN python -c "\
from pipespacing.detector import FINETUNED, GroundingDino; \
from pipespacing.depth import OUTDOOR, DepthAnythingMetric; \
FINETUNED.exists() or GroundingDino(device='cpu'); \
DepthAnythingMetric(device='cpu'); DepthAnythingMetric(OUTDOOR, device='cpu')"

EXPOSE 7860
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
