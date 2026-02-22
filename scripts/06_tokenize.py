#!/usr/bin/env python3
"""
Tokenize the deduplicated corpus and pack into fixed-length sequences for CPT.

Reads HF_TOKEN from the environment — never pass tokens as CLI arguments.
Saves a HuggingFace Arrow dataset to data/tokenized/cpt_dataset/.

Usage:
    HF_TOKEN=hf_... python scripts/06_tokenize.py \
        --input data/processed/corpus.jsonl \
        --model-id mistralai/Ministral-3-14B-Base-2512 \
        --seq-len 8192
"""
import argparse
import json
import logging
import os
import sys
from collections.abc import Generator, Iterator
from itertools import islice
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def iter_texts(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)["text"]


def batch_iter(iterable: Iterator[str], size: int) -> Generator[list[str], None, None]:
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch


def chunk_generator(
    jsonl_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
    batch_size: int = 2000,
) -> Generator[dict[str, list[int]], None, None]:
    """
    Stream-tokenize the corpus and yield packed, fixed-length sequences.

    Memory usage at any point is bounded by batch_size x avg_tokens_per_doc + seq_len,
    regardless of total corpus size.
    """
    eos_id: int | None = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(
            "Tokenizer has no eos_token_id. Set tokenizer.eos_token_id before tokenizing."
        )
    pad_id: int = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    buffer: list[int] = []
    doc_count = 0
    chunk_count = 0

    for batch in batch_iter(iter_texts(jsonl_path), batch_size):
        encoded: list[list[int]] = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        for ids in encoded:
            buffer.extend(ids)
            buffer.append(eos_id)
            doc_count += 1

        # Drain complete chunks immediately to keep memory bounded
        while len(buffer) >= seq_len:
            chunk = buffer[:seq_len]
            buffer = buffer[seq_len:]
            chunk_count += 1
            yield {
                "input_ids": chunk,
                "labels": chunk,
                "attention_mask": [1] * seq_len,
            }

        if doc_count % 50_000 == 0 and doc_count > 0:
            log.info("%d docs processed | %d chunks emitted", doc_count, chunk_count)

    # Yield final partial chunk if it contains enough content
    if len(buffer) > seq_len // 2:
        actual_len = len(buffer)
        padded = buffer + [pad_id] * (seq_len - actual_len)
        chunk_count += 1
        yield {
            "input_ids": padded[:seq_len],
            "labels": padded[:seq_len],
            # Attention mask is 0 for padding positions
            "attention_mask": [1] * actual_len + [0] * (seq_len - actual_len),
        }

    log.info(
        "Tokenization complete: %d docs -> %d chunks (seq_len=%d)",
        doc_count, chunk_count, seq_len,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize corpus for CPT")
    parser.add_argument("--input", default="data/processed/corpus.jsonl")
    parser.add_argument(
        "--model-id",
        default="mistralai/Ministral-3-14B-Base-2512",
        help="HuggingFace model ID for the tokenizer",
    )
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Documents per tokenizer call (trade memory vs speed, default: 2000)",
    )
    parser.add_argument("--output-dir", default="data/tokenized")
    # Token is read from the environment, not the CLI, to avoid exposure in process listings
    args = parser.parse_args()

    hf_token: str | None = os.environ.get("HF_TOKEN")
    if not hf_token:
        log.warning(
            "HF_TOKEN not set. This will fail if the model is gated. "
            "Set the environment variable: export HF_TOKEN=hf_..."
        )

    log.info("Loading tokenizer: %s", args.model_id)
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        args.model_id,
        token=hf_token,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    jsonl_path = Path(args.input)
    if not jsonl_path.exists():
        log.error("Input file not found: %s", jsonl_path)
        sys.exit(1)

    log.info(
        "Tokenizing: %s (seq_len=%d, batch_size=%d)",
        jsonl_path, args.seq_len, args.batch_size,
    )

    def _gen() -> Generator[dict[str, list[int]], None, None]:
        yield from chunk_generator(jsonl_path, tokenizer, args.seq_len, args.batch_size)

    ds: Dataset = Dataset.from_generator(_gen)

    out_dir = Path(args.output_dir) / "cpt_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    log.info("Dataset saved → %s  (%d examples)", out_dir, len(ds))


if __name__ == "__main__":
    main()
