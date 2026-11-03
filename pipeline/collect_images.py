"""
Step 1 — collect openly licensed photos of industrial piping. Writes images
to data/raw/images and an attribution CSV alongside (author + licence +
source URL for every file, as the CC licences require).

Sources (no API key needed for either):
  * openverse — Openverse search over Creative Commons photos (mostly Flickr).
                Anonymous access: 20 results/page, 240 per query, ~200 requests/day.
  * commons   — Wikimedia Commons. Fewer usable photos and an aggressive
                download rate limit, but useful as a top-up.

    python pipeline/collect_images.py --max 400
    python pipeline/collect_images.py --source commons --max 450   # top up
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterator

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw"
UA = os.environ.get("COLLECT_UA", "PipeSpacingDemo/0.1 (educational computer-vision portfolio project)")
MAX_SIDE = 1280

OPENVERSE_API = "https://api.openverse.org/v1/images/"
OPENVERSE_QUERIES = [
    "industrial piping", "factory pipes", "pipe rack", "industrial pipes",
    "refinery pipes", "pipework", "boiler room pipes", "plant room pipes",
    "chemical plant pipes", "power plant pipes", "pump room pipes",
    "mechanical room pipes", "brewery pipes", "steel pipes factory",
]

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
# Title must mention pipes; the extra words are matched anywhere in the page.
# (Plain full-text search drags in scanned books and anything with "pipe" in its description.)
COMMONS_QUERIES = [
    "intitle:pipework", "intitle:pipes industrial", "intitle:pipes plant",
    "intitle:pipes factory", "intitle:pipes station", "intitle:pipeline plant",
    "intitle:pipes refinery", "intitle:pipes boiler", "intitle:pipes pump",
    "intitle:piping industrial", "intitle:pipes mill", "intitle:pipes brewery",
    "intitle:pipes chemical", "intitle:\"pipe bridge\"", "intitle:pipes works",
]
COMMONS_THUMB = 960  # a standard Commons thumbnail size (others get rate-limited)

# Things that are also called "pipes" but are not what we want.
EXCLUDE = re.compile(
    r"organ|tobacco|smok|bagpipe|flute|pan ?pipe|pipe band|pipe dream|piping plover|"
    r"\bmap\b|diagram|drawing|logo|svg|patent|stamp|poster|museum|exhibit|"
    r"\(1[6-9]\d\d\)|volume|journal|magazine|\bpage\b|book",  # scanned old publications
    re.I,
)


def get(session: requests.Session, url: str, tries: int = 4, **kw) -> requests.Response:
    for attempt in range(tries):
        r = session.get(url, timeout=60, **kw)
        if r.status_code == 429:
            wait = min(int(r.headers.get("Retry-After", 0) or 5 * 2 ** attempt), 120)
            print(f"  rate limited, waiting {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"still rate limited: {url}")


# --- Sources: each yields {key, title, query, url, author, license, source} ---

def openverse(session: requests.Session) -> Iterator[dict]:
    for q in OPENVERSE_QUERIES:
        for page in range(1, 13):  # anonymous cap: 12 pages x 20
            try:
                data = get(session, OPENVERSE_API, params={
                    "q": q, "page": page, "page_size": 20,
                    "license_type": "all-cc", "category": "photograph",
                }).json()
            except Exception as e:
                print(f"  openverse {q!r} p{page}: {e}", file=sys.stderr)
                break
            for r in data.get("results", []):
                tags = " ".join(t["name"] for t in r.get("tags") or [])
                if EXCLUDE.search(f"{r.get('title', '')} {tags}"):
                    continue
                if (r.get("width") or MAX_SIDE) < 640:
                    continue
                yield {
                    "key": f"openverse:{r['id']}", "title": r.get("title") or r["id"], "query": q,
                    "url": r["url"], "author": r.get("creator") or "",
                    "license": f"CC {r['license'].upper()} {r.get('license_version') or ''}".strip(),
                    "source": r.get("foreign_landing_url") or r["url"],
                }
            if page >= data.get("page_count", 0):
                break
            time.sleep(1.0)


def commons(session: requests.Session) -> Iterator[dict]:
    def meta(info, key):
        raw = info.get("extmetadata", {}).get(key, {}).get("value", "")
        return re.sub(r"<[^>]+>", "", raw).strip()

    for q in COMMONS_QUERIES:
        params = {
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": f"{q} filetype:bitmap", "gsrnamespace": 6, "gsrlimit": 50,
            "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": COMMONS_THUMB,
        }
        while True:
            data = get(session, COMMONS_API, params=params).json()
            for page in data.get("query", {}).get("pages", {}).values():
                info = (page.get("imageinfo") or [{}])[0]
                licence = meta(info, "LicenseShortName")
                # Only originals wider than the thumbnail, so we always get a scaled thumb.
                if (EXCLUDE.search(page["title"]) or info.get("mime") not in ("image/jpeg", "image/png")
                        or info.get("width", 0) < COMMONS_THUMB or "thumburl" not in info or not licence):
                    continue
                yield {
                    "key": f"commons:{page['title']}", "title": page["title"], "query": q,
                    "url": info["thumburl"], "author": meta(info, "Artist"), "license": licence,
                    "source": info.get("descriptionurl", ""),
                }
            if "continue" not in data:
                break
            params.update(data["continue"])


SOURCES = {"openverse": openverse, "commons": commons}
FIELDS = ["file", "key", "title", "query", "author", "license", "source"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=SOURCES, default="openverse")
    ap.add_argument("--max", type=int, default=400, help="total images to have on disk")
    args = ap.parse_args()

    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    attr_path = OUT / "attribution.csv"
    rows = []
    if attr_path.exists():
        with open(attr_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    seen = {r["key"] for r in rows}
    hashes = {r["file"] for r in rows}

    session = requests.Session()
    session.headers["User-Agent"] = UA
    start = len(rows)

    try:
        for c in SOURCES[args.source](session):
            if len(rows) >= args.max:
                break
            if c["key"] in seen:
                continue
            seen.add(c["key"])
            try:
                content = get(session, c["url"]).content
                img = Image.open(io.BytesIO(content)).convert("RGB")
            except Exception as e:
                print(f"  skip {c['title'][:60]}: {e}", file=sys.stderr)
                continue
            if min(img.size) < 480:
                continue
            # Same photo is often indexed under several queries/sources: dedupe by content.
            name = hashlib.sha1(img.resize((32, 32)).tobytes()).hexdigest()[:12] + ".jpg"
            if name in hashes:
                continue
            hashes.add(name)
            img.thumbnail((MAX_SIDE, MAX_SIDE))
            img.save(img_dir / name, quality=92)
            rows.append({"file": name, **{k: c[k] for k in FIELDS[1:]}})
            print(f"[{len(rows)}/{args.max}] {c['query']!r}: {c['title'][:70]}", flush=True)
            time.sleep(0.3)
    finally:
        with open(attr_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    print(f"done: +{len(rows) - start} images, {len(rows)} total in {img_dir}")


if __name__ == "__main__":
    main()
