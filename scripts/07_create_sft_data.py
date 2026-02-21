#!/usr/bin/env python3
"""
Generate SFT (instruction-tuning) examples from the deduplicated corpus.

Uses template-based synthetic instructions — no external model calls required.
Templates:
  - Summarise: INPUT = full article (up to 4000 chars), OUTPUT = first paragraph
  - Title Q&A: INPUT = empty, OUTPUT = first paragraph (Wikipedia-style article start)
  - Continue: INPUT = first ~30% of article, OUTPUT = next 150 words

Produces: data/processed/sft_data.jsonl (Alpaca format: instruction / input / output)

Usage:
    python scripts/07_create_sft_data.py \
        --input data/processed/corpus.jsonl \
        --output data/processed/sft_data.jsonl \
        --max-docs 200000
"""
import argparse
import json
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── Templates ─────────────────────────────────────────────────────────────────

SUMMARIZE_TEMPLATES = [
    "Fasse den folgenden Text in wenigen Sätzen zusammen.",
    "Schreibe eine kurze Zusammenfassung des folgenden Abschnitts.",
    "Was sind die wichtigsten Punkte des folgenden Textes?",
    "Summarize the following passage in a few sentences.",
    "Provide a concise summary of the text below.",
]

TITLE_QA_TEMPLATES = [
    "Was ist {title}?",
    "Erkläre kurz, worum es sich bei {title} handelt.",
    "Gib einen Überblick über {title}.",
    "What is {title}?",
    "Give a brief overview of {title}.",
    "Describe {title} in a few sentences.",
]

CONTINUE_TEMPLATES = [
    "Vervollständige den folgenden Text sinnvoll.",
    "Schreibe den folgenden Abschnitt weiter.",
    "Continue the following passage coherently.",
    "Complete the following text:",
]


def _first_paragraph(text: str, min_chars: int = 80) -> str | None:
    """Return the first paragraph with at least *min_chars* characters."""
    for para in text.split("\n\n"):
        para = para.strip()
        if len(para) >= min_chars:
            return para
    return None


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    return cut[: last + 1] if last > max_chars // 2 else cut


def make_summary_example(doc: dict[str, object], rng: random.Random) -> dict[str, str] | None:
    """
    INPUT  = full article body (up to 4000 chars — longer than the output)
    OUTPUT = first paragraph only (≥80 chars), serving as a lead/abstract
    """
    text = str(doc.get("text", ""))
    summary = _first_paragraph(text)
    if not summary:
        return None

    # The input must be substantively longer than the output for summarisation to make sense
    body = _truncate_at_sentence(text, 4000)
    if len(body) <= len(summary) + 100:
        return None

    return {
        "instruction": rng.choice(SUMMARIZE_TEMPLATES),
        "input": body,
        "output": summary,
        "source": str(doc.get("source", "")),
        "lang": str(doc.get("lang", "")),
    }


def make_title_qa_example(doc: dict[str, object], rng: random.Random) -> dict[str, str] | None:
    """
    INPUT  = empty (the title carries the question)
    OUTPUT = first paragraph of the article (Wikipedia-style definition)
    """
    title = str(doc.get("title", "")).strip()
    if len(title) < 3:
        return None

    answer = _first_paragraph(str(doc.get("text", "")))
    if not answer:
        return None

    instruction = rng.choice(TITLE_QA_TEMPLATES).format(title=title)
    return {
        "instruction": instruction,
        "input": "",
        "output": _truncate_at_sentence(answer, 1000),
        "source": str(doc.get("source", "")),
        "lang": str(doc.get("lang", "")),
    }


def make_continue_example(doc: dict[str, object], rng: random.Random) -> dict[str, str] | None:
    """
    INPUT  = first ~25–40% of the article (prompt)
    OUTPUT = the following 150 words (natural continuation, non-overlapping)
    """
    text = str(doc.get("text", ""))
    words = text.split()
    if len(words) < 120:
        return None

    # Split point somewhere in the first quarter to first third
    split = rng.randint(len(words) // 5, len(words) // 3)
    prompt_text = " ".join(words[:split])
    continuation = " ".join(words[split : split + 150])

    return {
        "instruction": rng.choice(CONTINUE_TEMPLATES),
        "input": prompt_text,
        "output": continuation,
        "source": str(doc.get("source", "")),
        "lang": str(doc.get("lang", "")),
    }


GENERATORS = [make_summary_example, make_title_qa_example, make_continue_example]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SFT data from corpus")
    parser.add_argument("--input", default="data/processed/corpus.jsonl")
    parser.add_argument("--output", default="data/processed/sft_data.jsonl")
    parser.add_argument(
        "--samples-per-doc",
        type=int,
        default=1,
        help="Number of different templates to apply per document (max 3)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Maximum documents to process (0 = all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_file.with_suffix(".jsonl.tmp")

    written = 0
    processed = 0

    try:
        with open(args.input, encoding="utf-8") as inp, tmp.open("w", encoding="utf-8") as out_f:  # noqa: PTH123
            for line in inp:
                line = line.strip()
                if not line:
                    continue
                if args.max_docs and processed >= args.max_docs:
                    break
                doc: dict[str, object] = json.loads(line)
                processed += 1

                n = min(args.samples_per_doc, len(GENERATORS))
                chosen_generators = rng.sample(GENERATORS, k=n)

                for gen in chosen_generators:
                    example = gen(doc, rng)
                    if example is None:
                        continue
                    out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    written += 1

                if processed % 50_000 == 0:
                    log.info("%d docs processed | %d examples written", processed, written)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    tmp.rename(output_file)
    log.info("Done: %d docs → %d SFT examples → %s", processed, written, output_file)


if __name__ == "__main__":
    main()
