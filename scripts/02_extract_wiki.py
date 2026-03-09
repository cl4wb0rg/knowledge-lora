#!/usr/bin/env python3
"""
Extract clean article text from a Wikipedia XML dump using wikiextractor.

Produces: data/processed/wikipedia_{lang}.jsonl
Each line: {"id": ..., "title": ..., "url": ..., "text": ..., "source": "wikipedia", "lang": ...}

Usage:
    python scripts/02_extract_wiki.py --dump data/raw/dewiki-latest-pages-articles.xml.bz2 --lang de
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

LANG_RE = re.compile(r"^[a-z]{2,5}$")


def _validate_lang(lang: str) -> str:
    if not LANG_RE.match(lang):
        raise ValueError(f"Invalid language code {lang!r}")
    return lang


def extract_with_wikiextractor(dump_path: Path, tmp_dir: Path, lang: str) -> Path:
    out_dir = tmp_dir / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    n_workers = os.cpu_count() or 4

    cmd = [
        "wikiextractor",
        str(dump_path),
        "--output",
        str(out_dir),
        "--json",
        "--quiet",
        "--no-templates",
        "--processes",
        str(n_workers),
    ]
    log.info("Running wikiextractor with %d workers: %s", n_workers, " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_dir


def merge_shards(wikiextractor_out: Path, dest_file: Path, lang: str) -> int:
    """Merge all wikiextractor shard files into a single JSONL (atomic write)."""
    tmp = dest_file.with_suffix(".jsonl.tmp")
    count = 0

    try:
        with tmp.open("w", encoding="utf-8") as out_f:
            # Sort by numeric shard index embedded in the filename for stable ordering
            shards = sorted(
                wikiextractor_out.rglob("wiki_*"),
                key=lambda p: (p.parent.name, p.name),
            )
            for shard in shards:
                with shard.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        doc = json.loads(line)
                        text = doc.get("text", "").strip()
                        if len(text) < 200:
                            continue
                        record = {
                            "id": doc.get("id"),
                            "title": doc.get("title"),
                            "url": doc.get("url"),
                            "text": text,
                            "source": "wikipedia",
                            "lang": lang,
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    tmp.rename(dest_file)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Wikipedia XML dump to JSONL")
    parser.add_argument("--dump", required=True, help="Path to .xml.bz2 dump file")
    parser.add_argument("--lang", default="de", help="Language code (2-5 lowercase letters)")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument(
        "--tmp-dir",
        default="data/raw/wikiextractor_tmp",
        help="Temp directory for wikiextractor shards",
    )
    args = parser.parse_args()

    _validate_lang(args.lang)
    dump_path = Path(args.dump)
    if not dump_path.exists():
        log.error("Dump not found: %s", dump_path)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(args.tmp_dir)
    wikiextractor_out = extract_with_wikiextractor(dump_path, tmp_dir, args.lang)

    dest = out_dir / f"wikipedia_{args.lang}.jsonl"
    count = merge_shards(wikiextractor_out, dest, args.lang)
    log.info("Wrote %d articles → %s", count, dest)


if __name__ == "__main__":
    main()
