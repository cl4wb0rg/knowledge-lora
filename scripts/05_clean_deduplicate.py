#!/usr/bin/env python3
"""
Merge all JSONL sources, clean text, and deduplicate.

Two-stage deduplication:
  1. Exact dedup via SHA-256 of normalised text
  2. Near-dedup via MinHash LSH (Jaccard similarity on word 5-grams)

Produces: data/processed/corpus.jsonl

Usage:
    python scripts/05_clean_deduplicate.py \
        --input-files data/processed/wikipedia_de.jsonl \
                      data/processed/pdfs.jsonl \
                      data/processed/markdown.jsonl \
        --output-file data/processed/corpus.jsonl
"""

import argparse
import hashlib
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

from datasketch import MinHash, MinHashLSH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_shingles(text: str, k: int = 5) -> set[str]:
    words = text.split()
    if len(words) <= k:
        return {text}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def build_minhash(text: str, num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for shingle in word_shingles(normalize(text)):
        m.update(shingle.encode("utf-8"))
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge + deduplicate JSONL corpora")
    parser.add_argument(
        "--input-files",
        nargs="+",
        required=True,
        help="JSONL files to process (processed in order; earlier files take priority)",
    )
    parser.add_argument("--output-file", default="data/processed/corpus.jsonl")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="MinHash Jaccard similarity threshold for near-dedup (default: 0.8)",
    )
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--min-length", type=int, default=200)
    args = parser.parse_args()

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_file.with_suffix(".jsonl.tmp")

    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)
    seen_hashes: set[str] = set()
    # Use a monotonic counter as the LSH key to avoid hash collisions
    lsh_key_counter = 0

    total = kept = dupes = too_short = 0

    try:
        with tmp.open("w", encoding="utf-8") as out_f:
            for src_path in args.input_files:
                log.info("Processing: %s", src_path)
                with open(src_path, encoding="utf-8") as f:  # noqa: PTH123
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        total += 1

                        doc = json.loads(line)
                        text = doc.get("text", "").strip()

                        if len(text) < args.min_length:
                            too_short += 1
                            continue

                        # Stage 1: exact dedup via SHA-256
                        norm = normalize(text)
                        sha = hashlib.sha256(norm.encode()).hexdigest()
                        if sha in seen_hashes:
                            dupes += 1
                            continue
                        seen_hashes.add(sha)

                        # Stage 2: near-dedup via MinHash LSH
                        mh = build_minhash(text, args.num_perm)
                        if lsh.query(mh):
                            dupes += 1
                            continue

                        lsh_key = str(lsh_key_counter)
                        lsh.insert(lsh_key, mh)
                        lsh_key_counter += 1

                        out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                        kept += 1

                        if total % 10_000 == 0:
                            log.info("%10d read | %10d kept | %8d dupes", total, kept, dupes)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    tmp.rename(output_file)
    log.info("─" * 55)
    log.info("Total read   : %10d", total)
    log.info("Too short    : %10d", too_short)
    log.info("Duplicates   : %10d", dupes)
    log.info("Kept         : %10d", kept)
    log.info("Output       : %s", output_file)


if __name__ == "__main__":
    main()
