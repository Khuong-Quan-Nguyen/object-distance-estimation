"""
Batch scan: run every image in a folder through detection + depth + spacing,
and write an inspection report.

    python scan.py data/scan_input --od-mm 114.3
    python scan.py photos/ --depth-dir depth_frames/     # real RGB-D frames
    python scan.py photos/ --mode reference --od-mm 168.3  # no depth model

Outputs (in --out, default runs/scan):
    annotated/<image>.jpg   boxes, pipe IDs, dimension lines (red = too tight)
    pipes.csv               one row per detected pipe
    pairs.csv               one row per neighbouring pipe pair
    summary.json            per-image counts + flagged pairs
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from PIL import Image

from pipespacing.analyze import analyze
from pipespacing.depth import INDOOR, OUTDOOR, DepthAnythingMetric, load_depth_file
from pipespacing.detector import load_detector

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="folder of images (or one image)")
    ap.add_argument("--out", type=Path, default=Path("runs/scan"))
    ap.add_argument("--mode", choices=["auto", "depth", "reference"], default="auto",
                    help="auto = fused when --od-mm is given, else depth")
    ap.add_argument("--od-mm", type=float, default=None,
                    help="known pipe outside diameter, e.g. 114.3 for 4in NPS")
    ap.add_argument("--depth-dir", type=Path, default=None,
                    help="aligned depth frames named like the images (.png mm or .npy m)")
    ap.add_argument("--outdoor", action="store_true", help="use the outdoor depth model")
    ap.add_argument("--focal-px", type=float, default=None)
    ap.add_argument("--hfov", type=float, default=65.0, help="assumed HFOV when no EXIF")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--min-clearance-mm", type=float, default=25.0)
    ap.add_argument("--weights", type=Path, default=None)
    args = ap.parse_args()

    if args.mode == "reference" and not args.od_mm:
        ap.error("--mode reference needs --od-mm")

    files = [args.input] if args.input.is_file() else sorted(
        p for p in args.input.rglob("*") if p.suffix.lower() in IMAGE_EXT
    )
    if not files:
        ap.error(f"no images found in {args.input}")

    detector = load_detector(args.weights)
    depth_fn = None
    if args.mode != "reference" and not args.depth_dir:
        depth_fn = DepthAnythingMetric(OUTDOOR if args.outdoor else INDOOR)
    print(f"detector: {detector.source} | {len(files)} images")

    (args.out / "annotated").mkdir(parents=True, exist_ok=True)
    pipe_rows, pair_rows, summary = [], [], []
    t0 = time.time()

    for i, path in enumerate(files, 1):
        rel = path.relative_to(args.input) if args.input.is_dir() else Path(path.name)
        depth_map = None
        if args.depth_dir:
            for ext in (".npy", ".png"):
                cand = (args.depth_dir / rel).with_suffix(ext)
                if cand.exists():
                    depth_map = load_depth_file(cand)
                    break
            else:
                print(f"  {rel}: no depth frame, using reference geometry")

        a = analyze(
            Image.open(path), detector, depth_fn=depth_fn, depth_map=depth_map,
            od_mm=args.od_mm, focal_px=args.focal_px, hfov_deg=args.hfov,
            conf=args.conf, min_clearance_mm=args.min_clearance_mm,
        )
        out_name = str(rel.with_suffix(".jpg")).replace("\\", "__").replace("/", "__")
        a.annotated.save(args.out / "annotated" / out_name, quality=90)

        for p in a.pipes:
            pipe_rows.append({
                "image": str(rel), "pipe": p.id, "orientation": p.orientation,
                "confidence": round(p.confidence, 3), "diameter_px": round(p.diameter_px, 1),
                "depth_m": p.depth_m, "od_mm": p.od_mm, "nps": p.nps,
                "x1": round(p.box[0]), "y1": round(p.box[1]), "x2": round(p.box[2]), "y2": round(p.box[3]),
            })
        for pr in a.pairs:
            pair_rows.append({"image": str(rel), "mode": a.mode, **pr.to_dict()})
        flagged = [pr.to_dict() for pr in a.pairs if pr.flagged]
        summary.append({"image": str(rel), "mode": a.mode, "pipes": len(a.pipes),
                        "pairs": len(a.pairs), "flagged": flagged})

        tag = f"  !! {len(flagged)} tight" if flagged else ""
        print(f"[{i}/{len(files)}] {rel}: {len(a.pipes)} pipes, {len(a.pairs)} pairs{tag}")

    write_csv(args.out / "pipes.csv", pipe_rows)
    write_csv(args.out / "pairs.csv", pair_rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    n_flag = sum(len(s["flagged"]) for s in summary)
    dt = time.time() - t0
    print(f"\n{len(files)} images in {dt:.1f}s ({dt / len(files):.2f}s/img) — "
          f"{len(pipe_rows)} pipes, {len(pair_rows)} pairs, {n_flag} below "
          f"{args.min_clearance_mm:.0f} mm clearance\nreport: {args.out.resolve()}")


def write_csv(path: Path, rows: list):
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
