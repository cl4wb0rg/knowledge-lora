#!/usr/bin/env python3
"""
Extract plain text from Markdown / reStructuredText / plain-text files.

Note: absolute file paths are NOT stored to avoid leaking host directory structure.

Produces: data/processed/markdown.jsonl
Each line: {"id": ..., "title": ..., "text": ..., "source": "markdown"}

Usage:
    python scripts/04_extract_markdown.py --input-dir /path/to/docs
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

EXTENSIONS = {".md", ".markdown", ".txt", ".rst"}


def markdown_to_text(raw: str) -> str:
    # Remove fenced code blocks
    raw = re.sub(r"```[\s\S]*?```", "", raw)
    raw = re.sub(r"~~~[\s\S]*?~~~", "", raw)
    # Remove inline code
    raw = re.sub(r"`[^`\n]+`", "", raw)
    # Remove images
    raw = re.sub(r"!\[.*?\]\(.*?\)", "", raw)
    # Convert links to label text only
    raw = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", raw)
    # Remove HTML tags
    raw = re.sub(r"<[^>]+>", "", raw)
    # Remove Markdown headings (keep text)
    raw = re.sub(r"^#{1,6}\s+", "", raw, flags=re.MULTILINE)
    # Remove setext headings (lines of = or - that follow a text line)
    raw = re.sub(r"^[=]{3,}\s*$", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"^[-]{3,}\s*$", "", raw, flags=re.MULTILINE)
    # Remove bold/italic with matching delimiters (*** ** * ___ __ _)
    # Use backreference to ensure opening and closing delimiter lengths match
    raw = re.sub(r"(\*{1,3})((?:(?!\1)[\s\S])+?)\1", r"\2", raw)
    raw = re.sub(r"(_{1,3})((?:(?!\1)[\s\S])+?)\1", r"\2", raw)
    # Remove horizontal rules (three or more of the same char on a line alone)
    raw = re.sub(r"^[*_]{3,}\s*$", "", raw, flags=re.MULTILINE)
    # Remove blockquote markers
    raw = re.sub(r"^>\s?", "", raw, flags=re.MULTILINE)
    # Remove RST directives (.. something::)
    raw = re.sub(r"^\.\. \w[\w-]*::", "", raw, flags=re.MULTILINE)
    # Collapse blank lines
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text from Markdown/text files")
    parser.add_argument("--input-dir", required=True, help="Directory to scan (recursive)")
    parser.add_argument("--output-file", default="data/processed/markdown.jsonl")
    parser.add_argument(
        "--min-length",
        type=int,
        default=200,
        help="Minimum character count to keep a document (default: 200)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        log.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_file.with_suffix(".jsonl.tmp")

    files = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in EXTENSIONS)
    log.info("Found %d files in %s", len(files), input_dir)

    kept = 0
    skipped = 0
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for file_path in files:
                try:
                    raw = file_path.read_text(encoding="utf-8", errors="replace")
                    text = markdown_to_text(raw)
                    if len(text) < args.min_length:
                        skipped += 1
                        continue
                    record = {
                        "id": str(file_path.relative_to(input_dir)),
                        "title": file_path.stem.replace("_", " ").replace("-", " "),
                        "text": text,
                        "source": "markdown",
                        # Relative path only — avoids leaking host filesystem layout
                        "file": str(file_path.relative_to(input_dir)),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept += 1
                except Exception as exc:
                    log.error("Failed to process %s: %s", file_path, exc, exc_info=False)
                    skipped += 1
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    tmp.rename(output_file)
    log.info("Kept %d | Skipped %d → %s", kept, skipped, output_file)


if __name__ == "__main__":
    main()
