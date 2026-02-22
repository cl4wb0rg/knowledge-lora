#!/usr/bin/env python3
"""
Generate high-quality Q&A pairs from the deduplicated corpus using a local LLM
served via vLLM (offline batch inference — no external API calls).

Intended workflow:
  1. Run CPT training (train_cpt.sh)
  2. Merge CPT LoRA adapter:
       accelerate launch -m axolotl.cli.merge_lora configs/cpt_config.yaml \\
           --lora-model-dir output/cpt/checkpoint-final
  3. Run this script pointing at the merged model:
       python scripts/08_generate_qa_llm.py \\
           --model output/cpt/merged \\
           --input data/processed/corpus.jsonl \\
           --output data/processed/sft_qa_llm.jsonl

The base model (without CPT) also works but will produce lower-quality Q&A for
documents outside its training data.

Output format: Alpaca JSONL (instruction / input / output) — compatible with
07_create_sft_data.py output and axolotl SFT config (type: alpaca).

Usage:
    python scripts/08_generate_qa_llm.py \\
        --model mistralai/Ministral-3-14B-Base-2512 \\
        --input data/processed/corpus.jsonl \\
        --output data/processed/sft_qa_llm.jsonl \\
        --max-docs 100000 \\
        --qa-per-doc 3 \\
        --batch-size 64
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

# ── Prompt templates ──────────────────────────────────────────────────────────

# The model is prompted to produce a numbered list of Q&A pairs.
# Keeping the format simple makes parsing robust.
PROMPT_DE = """\
Lies den folgenden Text sorgfältig und erstelle daraus {n} Frage-Antwort-Paare auf Deutsch.
Jedes Paar muss direkt aus dem Text beantwortbar sein.

Format (exakt einhalten):
F1: <Frage>
A1: <Antwort>
F2: <Frage>
A2: <Antwort>

Text:
{text}

Frage-Antwort-Paare:"""

PROMPT_EN = """\
Read the following text carefully and create {n} question-answer pairs in English.
Each answer must be directly supported by the text.

Format (follow exactly):
Q1: <question>
A1: <answer>
Q2: <question>
A2: <answer>

Text:
{text}

Question-answer pairs:"""

# Simple heuristic: use German prompt for 'de' docs, English for everything else
def _pick_prompt(lang: str, text: str, n: int) -> str:
    excerpt = text[:2000].strip()
    if lang == "de":
        return PROMPT_DE.format(n=n, text=excerpt)
    return PROMPT_EN.format(n=n, text=excerpt)


# ── Output parsing ────────────────────────────────────────────────────────────

# Matches "F1: ..." / "Q1: ..." style lines (German and English)
_QA_PATTERN = re.compile(
    r"[FQ](\d+):\s*(.+?)\s*\n[AB]\1:\s*(.+?)(?=\n[FQ]\d+:|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def _parse_qa(raw: str, source: str, lang: str) -> list[dict[str, str]]:
    """Extract (instruction=question, input='', output=answer) dicts from raw text."""
    results = []
    for match in _QA_PATTERN.finditer(raw):
        question = match.group(2).strip()
        answer = match.group(3).strip()
        if len(question) < 10 or len(answer) < 10:
            continue
        results.append(
            {
                "instruction": question,
                "input": "",
                "output": answer,
                "source": source,
                "lang": lang,
            }
        )
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LLM-based Q&A pairs from corpus.jsonl using vLLM"
    )
    parser.add_argument(
        "--model",
        default="mistralai/Ministral-3-14B-Base-2512",
        help="HuggingFace model ID or local path to (merged) model weights",
    )
    parser.add_argument("--input", default="data/processed/corpus.jsonl")
    parser.add_argument("--output", default="data/processed/sft_qa_llm.jsonl")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Maximum documents to process (0 = all)",
    )
    parser.add_argument(
        "--qa-per-doc",
        type=int,
        default=3,
        help="Number of Q&A pairs to request per document (1-5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of prompts submitted to vLLM at once",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=768,
        help="Maximum tokens the model may generate per prompt",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--min-doc-chars",
        type=int,
        default=300,
        help="Skip documents shorter than this many characters",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (1 for DGX Spark single-GPU)",
    )
    args = parser.parse_args()

    args.qa_per_doc = max(1, min(5, args.qa_per_doc))

    # Late import — vLLM is only needed at inference time
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        log.error("vLLM not installed. Run: pip install vllm")
        sys.exit(1)

    log.info("Loading model: %s", args.model)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=False,
        max_model_len=4096,  # prompt + generation; raise if OOM allows
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        stop=["</s>", "[INST]", "\n\nText:"],  # prevent prompt repetition
    )
    log.info("Model loaded.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".jsonl.tmp")

    # Read corpus into batches
    docs_seen = 0
    written = 0

    try:
        with (
            Path(args.input).open(encoding="utf-8") as inp,
            tmp_path.open("w", encoding="utf-8") as out_f,
        ):
            batch_docs: list[dict] = []
            batch_prompts: list[str] = []

            def _flush_batch() -> None:
                nonlocal written
                if not batch_prompts:
                    return
                outputs = llm.generate(batch_prompts, sampling_params)
                for doc, result in zip(batch_docs, outputs, strict=True):
                    raw = result.outputs[0].text
                    pairs = _parse_qa(
                        raw,
                        source=str(doc.get("source", "")),
                        lang=str(doc.get("lang", "")),
                    )
                    for pair in pairs:
                        out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                        written += 1
                batch_docs.clear()
                batch_prompts.clear()

            for line in inp:
                line = line.strip()
                if not line:
                    continue
                if args.max_docs and docs_seen >= args.max_docs:
                    break

                doc: dict = json.loads(line)
                text = str(doc.get("text", ""))
                if len(text) < args.min_doc_chars:
                    continue

                docs_seen += 1
                lang = str(doc.get("lang", ""))
                prompt = _pick_prompt(lang, text, args.qa_per_doc)
                batch_docs.append(doc)
                batch_prompts.append(prompt)

                if len(batch_prompts) >= args.batch_size:
                    _flush_batch()
                    log.info(
                        "%d docs processed | %d Q&A pairs written",
                        docs_seen,
                        written,
                    )

            _flush_batch()  # remaining partial batch

    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    tmp_path.rename(output_path)
    log.info(
        "Done: %d docs → %d Q&A pairs → %s",
        docs_seen,
        written,
        output_path,
    )


if __name__ == "__main__":
    main()
