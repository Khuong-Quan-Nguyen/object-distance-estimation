"""
Step 2 — auto-label. Run zero-shot YOLO-World ("pipe" prompts) over the raw
images and store draft boxes in data/labels.json for human review
(frontend/review.html). Images that were already reviewed are left alone.

    python pipeline/autolabel.py --conf 0.15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipespacing.detector import GroundingDino, PipeDetector  # noqa: E402
from pipespacing.labels import IMAGES, load_labels, save_labels  # noqa: E402


def keep(box, w, h, min_side=8, max_area=0.6):
    """Drop slivers and 'whole scene' boxes that zero-shot models like to emit."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return min(bw, bh) >= min_side and (bw * bh) / (w * h) <= max_area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeler", choices=["gdino", "yoloworld", "finetuned"], default="gdino",
                    help="finetuned = your trained model, for a second labelling round")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--redo", action="store_true", help="also overwrite unreviewed drafts")
    args = ap.parse_args()

    detector = {
        "gdino": GroundingDino,
        "yoloworld": lambda: PipeDetector(zero_shot=True),
        "finetuned": PipeDetector,
    }[args.labeler]()
    print(f"labeler: {detector.source}")
    labels = load_labels()
    files = sorted(IMAGES.glob("*.jpg"))
    stats = {"labelled": 0, "boxes": 0, "skipped": 0}

    for i, path in enumerate(files, 1):
        entry = labels.get(path.name)
        reviewed = entry and entry["status"] != "pending" and not entry.get("auto_rejected")
        if entry and (reviewed or not args.redo):
            stats["skipped"] += 1
            continue
        img = Image.open(path).convert("RGB")
        boxes, confs = detector(img, conf=args.conf)
        draft = [
            [round(v, 1) for v in b] + [round(c, 3)]
            for b, c in zip(boxes, confs)
            if keep(b, *img.size)
        ]
        # Web search drags in off-topic photos; no pipe found at all is a strong hint.
        # They land in the "rejected" tab of the review UI, so a human can still rescue one.
        labels[path.name] = {
            "status": "pending" if draft else "rejected",
            "auto_rejected": not draft,
            "width": img.width, "height": img.height,
            "boxes": [b[:4] for b in draft],
            "auto": draft,
        }
        stats["labelled"] += 1
        stats["boxes"] += len(draft)
        if i % 25 == 0:
            save_labels(labels)
            print(f"{i}/{len(files)}")

    save_labels(labels)
    print(f"auto-labelled {stats['labelled']} images ({stats['boxes']} boxes), "
          f"kept {stats['skipped']} existing — review at http://localhost:8010/review.html")


if __name__ == "__main__":
    main()
