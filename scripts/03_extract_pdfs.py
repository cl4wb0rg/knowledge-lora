#!/usr/bin/env python3
"""
Extract text from PDF files using PyMuPDF (fitz).

Produces: data/processed/pdfs.jsonl
Each line: {"id": ..., "title": ..., "text": ..., "source": "pdf"}
Note: absolute file paths are NOT stored to avoid leaking host directory structure.

Usage:
    python scripts/03_extract_pdfs.py --input-dir /path/to/pdfs
"""
import argparse
import json
import logging
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract all text from *pdf_path*, closing the document even on error."""
    pages: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text: str = page.get_text("text")
            if text.strip():
                pages.append(text)
    return "\n\n".join(pages)


def clean_pdf_text(text: str) -> str:
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces (but not newlines)
    text = re.sub(r"[^\S\n]{2,}", " ", text)
    # Remove isolated page-number lines (digits / roman numerals only)
    lines = [ln for ln in text.splitlines() if not re.fullmatch(r"\s*[\divxlcIVXLC]+\s*", ln)]
    return "\n".join(lines).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text from PDFs")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory to scan for .pdf files (recursive)",
    )
    parser.add_argument("--output-file", default="data/processed/pdfs.jsonl")
    parser.add_argument(
        "--min-length",
        type=int,
        default=500,
        help="Minimum character count to keep a document (default: 500)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        log.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_file.with_suffix(".jsonl.tmp")

    pdfs = sorted(input_dir.rglob("*.pdf"))
    log.info("Found %d PDF files in %s", len(pdfs), input_dir)

    kept = 0
    skipped = 0
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for pdf_path in pdfs:
                try:
                    raw = extract_pdf_text(pdf_path)
                    text = clean_pdf_text(raw)
                    if len(text) < args.min_length:
                        skipped += 1
                        continue
                    record = {
                        "id": pdf_path.stem,
                        "title": pdf_path.stem.replace("_", " ").replace("-", " "),
                        "text": text,
                        "source": "pdf",
                        # Store relative path to avoid leaking host directory structure
                        "file": str(pdf_path.relative_to(input_dir)),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept += 1
                except Exception as exc:
                    log.error("Failed to extract %s: %s", pdf_path.name, exc, exc_info=False)
                    skipped += 1
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    tmp.rename(output_file)
    log.info("Kept %d | Skipped %d → %s", kept, skipped, output_file)


if __name__ == "__main__":
    main()
