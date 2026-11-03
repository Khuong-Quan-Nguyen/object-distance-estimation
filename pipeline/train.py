"""
Step 3 — fine-tune a single-class YOLOv8n "pipe" detector on reviewed labels.

Exports data/labels.json to YOLO format (deterministic 80/20 split by file
hash so the val set is stable across runs), trains, and copies the best
weights to models/pipe_yolov8n.pt where the scanner and API pick them up.

    python pipeline/train.py --epochs 60
    python pipeline/train.py --include-pending   # quick run on unreviewed drafts
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipespacing.detector import FINETUNED, MODELS  # noqa: E402
from pipespacing.labels import DATA, IMAGES, load_labels  # noqa: E402

DATASET = DATA / "dataset"


def split_of(name: str, val_frac: float) -> str:
    h = int(hashlib.md5(name.encode()).hexdigest(), 16) % 1000
    return "val" if h < val_frac * 1000 else "train"


def export(include_pending: bool, val_frac: float) -> dict:
    labels = load_labels()
    wanted = {"accepted", "pending"} if include_pending else {"accepted"}
    if DATASET.exists():
        shutil.rmtree(DATASET)
    counts = {"train": 0, "val": 0, "boxes": 0}
    for name, e in labels.items():
        if e["status"] not in wanted:
            continue
        split = split_of(name, val_frac)
        img_dir, lbl_dir = DATASET / "images" / split, DATASET / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(IMAGES / name, img_dir / name)
        W, H = e["width"], e["height"]
        lines = []
        for x1, y1, x2, y2 in e["boxes"]:
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        # An empty label file is a valid "no pipes here" background image.
        (lbl_dir / name).with_suffix(".txt").write_text("\n".join(lines))
        counts[split] += 1
        counts["boxes"] += len(lines)

    (DATASET / "data.yaml").write_text(
        f"path: {DATASET.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: pipe\n"
    )
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8, help="8 fits a 4 GB GPU at 640px")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--include-pending", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    counts = export(args.include_pending, args.val_frac)
    print(f"dataset: {counts['train']} train / {counts['val']} val images, {counts['boxes']} boxes")
    if counts["train"] == 0 or counts["val"] == 0:
        sys.exit("not enough labelled images — review some in review.html, or pass --include-pending")

    import torch
    from ultralytics import YOLO

    MODELS.mkdir(exist_ok=True)
    model = YOLO(str(MODELS / "yolov8n.pt"))
    model.train(
        data=str(DATASET / "data.yaml"),
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device or (0 if torch.cuda.is_available() else "cpu"),
        project=str(ROOT / "runs" / "train"), name="pipe", exist_ok=True,
        workers=2, patience=20, plots=True,
        # Pipe racks are often shot at an angle, and horizontal/vertical runs both occur.
        degrees=5, fliplr=0.5, flipud=0.1,
    )
    best = ROOT / "runs" / "train" / "pipe" / "weights" / "best.pt"
    shutil.copy2(best, FINETUNED)
    metrics = YOLO(str(FINETUNED)).val(data=str(DATASET / "data.yaml"), split="val",
                                      project=str(ROOT / "runs" / "val"), exist_ok=True)
    print(f"saved {FINETUNED}  |  val mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
