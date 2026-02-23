# knowledge-lora

Continued Pretraining (CPT) + Supervised Fine-Tuning (SFT) LoRA adapter for
**[Ministral-3-14B-Base-2512](https://huggingface.co/mistralai/Ministral-3-14B-Base-2512)**,
trained on German Wikipedia and custom documents (PDFs, Markdown) to extend the
model's knowledge cutoff.

## Pipeline run status

| Step | Status | Output | Notes |
|------|--------|--------|-------|
| 01 download | ✅ done | `data/raw/dewiki-latest-pages-articles.xml.bz2` (7.3 GB) | |
| 02 extract | ✅ done | `data/processed/wikipedia_de.jsonl` (7.6 GB, 2,625,635 articles) | |
| 03 PDF extract | — skipped | — | no PDF inputs |
| 04 Markdown extract | — skipped | — | no MD inputs |
| 05 dedup | ✅ done | `data/processed/corpus.jsonl` (7.6 GB, 2,614,035 articles) | 11,600 dupes removed |
| 06 tokenize | ✅ done | `data/tokenized/cpt_dataset/` (236963 sequences) | Ministral tokenizer, seq_len 8192 |
| 07 SFT data | ✅ done | `data/processed/sft_data.jsonl` (2.2 GB, 1,543,744 examples) | template-based |
| 08 QA generation | ⏳ pending | `data/processed/sft_qa_llm.jsonl` | runs after CPT merge |
| CPT training | 🔄 running | `output/cpt/` | axolotl, LoRA rank 128, 2400 steps (~77k seqs, ~7 days); micro_batch=4, grad_ckpt=off |
| SFT training | ⏳ pending | `output/sft/` | axolotl, LoRA rank 64 |

## Architecture overview

```
Wikipedia dump ──┐
PDF files        ├── Data Pipeline ──► deduplicated corpus ──► CPT LoRA ──► SFT LoRA
Markdown files ──┘                       (corpus.jsonl)         (rank 128)   (rank 64)
```

**Phase 1 – CPT:** The base model learns new factual knowledge via next-token prediction
on the cleaned corpus (LoRA rank 128, full BF16, seq_len 8192).

**Phase 2 – SFT:** The CPT adapter is used as a starting point for instruction-following
fine-tuning on template-generated Q&A / summarisation data (LoRA rank 64, lr 2e-5).

## Hardware requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM / unified memory | 24 GB (with QLoRA) | 128 GB (DGX Spark / GB10) |
| Storage for German Wikipedia | ~25 GB raw | ~60 GB with intermediates |
| Python | 3.11+ | 3.11+ |

Training was designed for the **Dell GB10 / NVIDIA DGX Spark** (128 GB unified
Blackwell memory) — full BF16 with no quantisation.

## Installation

```bash
# 1. Clone
git clone https://github.com/cl4wb0rg/knowledge-lora.git
cd knowledge-lora

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install all training dependencies (handles CUDA 13.0 / GB10 quirks)
bash install.sh

# 4. Configure credentials
cp .env.example .env
# Edit .env and add your HuggingFace token
```

> **DGX Spark / CUDA 13.0 notes:**
> - `install.sh` handles the correct installation order for torch, axolotl (GitHub HEAD), flash-attn, and xformers.
> - **xformers** cannot be built for CUDA 13.0 (removed driver API symbols) and is skipped. `install.sh` uses `--only-binary xformers` so pip never attempts a source compile — previously, the source build spawned many parallel `nvcc`/`cicc` processes that exhausted all RAM+swap and froze the system. flash-attn covers the same functionality.
> - **flash-attn** is built from source against CUDA 13.0 (no pre-built cu13 wheel exists). The build uses `MAX_JOBS=1` and `nice -n 10` to stay below 80 % CPU/RAM load. **Expect ~20–30 minutes on first install**; subsequent runs skip the build if flash-attn is already correctly installed.
> - **GB10 freeze prevention:** before each flash-attn build, `install.sh` clears the filesystem cache and caps GPU power to 80 % of its maximum (via `nvidia-smi -pl`) to prevent hard crashes from memory exhaustion or power spikes during nvcc compilation. Requires `sudo`; silently skipped if unavailable. The power limit is restored after the build completes. The flash-attn build subprocess also sets a high OOM score (`oom_score_adj=500`) so the kernel kills it first if memory runs out, keeping the system responsive.
> - axolotl is installed from GitHub HEAD; PyPI releases do not support torch 2.10+.

### Verify the installation

After `install.sh` completes, run the smoke test to confirm the full training
stack (torch CUDA, flash-attn, LoRA, forward/backward pass) works end-to-end.
No large model download or pre-processed data needed — it uses `gpt2` (124 M)
with synthetic inputs.

```bash
python scripts/smoke_test.py
# Expected output:
#   [OK]  torch 2.10.0+cu130  |  NVIDIA GB10  |  120 GB
#   [OK]  flash-attn 2.8.3
#   [OK]  gpt2 loaded on cuda (bf16)
#   [OK]  LoRA applied  |  trainable 147.5K / 124.6M params
#     step 1/5  loss=...
#     ...
#   [OK]  all losses finite  |  first=...  last=...
#   === PASSED ===
```

### vLLM inference environment (optional)

vLLM requires a different torch version than axolotl and must live in its own venv:

```bash
bash install_vllm.sh        # creates .venv-vllm/
source .venv-vllm/bin/activate
```

Use this environment only for running `scripts/08_generate_qa_llm.py` after CPT.

## Data pipeline

Run the steps in order. All steps are idempotent — re-running skips already
completed work where possible.

### Step 1 — Download Wikipedia

```bash
python scripts/01_download_wiki.py --lang de
# For multiple languages:
python scripts/01_download_wiki.py --lang de --lang en --lang fr
```

Downloads `data/raw/dewiki-latest-pages-articles.xml.bz2` and verifies the MD5
checksum against Wikimedia's published checksums. The download is **resumable**
— if the connection drops, re-running the script continues from where it left off.

### Step 2 — Extract Wikipedia text

```bash
python scripts/02_extract_wiki.py \
    --dump data/raw/dewiki-latest-pages-articles.xml.bz2 \
    --lang de
```

Produces `data/processed/wikipedia_de.jsonl`.
Uses all available CPU cores by default.

### Step 3 — Extract PDFs (optional)

```bash
python scripts/03_extract_pdfs.py --input-dir /path/to/your/pdfs
```

Produces `data/processed/pdfs.jsonl`.

### Step 4 — Extract Markdown / text files (optional)

```bash
python scripts/04_extract_markdown.py --input-dir /path/to/your/docs
```

Supports `.md`, `.markdown`, `.txt`, `.rst`.
Produces `data/processed/markdown.jsonl`.

### Step 5 — Deduplicate

```bash
python scripts/05_clean_deduplicate.py \
    --input-files \
        data/processed/wikipedia_de.jsonl \
        data/processed/pdfs.jsonl \
        data/processed/markdown.jsonl \
    --output-file data/processed/corpus.jsonl
```

Two-stage deduplication: exact (SHA-256) + near-duplicate (MinHash LSH, Jaccard ≥ 0.8).

### Step 6 — Tokenize

```bash
HF_TOKEN=hf_... python scripts/06_tokenize.py \
    --input data/processed/corpus.jsonl \
    --model-id mistralai/Ministral-3-14B-Base-2512 \
    --seq-len 8192
```

Produces a packed Arrow dataset at `data/tokenized/cpt_dataset/`.
Memory usage is bounded by `--batch-size` regardless of corpus size.

### Step 7 — Generate SFT data (template-based)

```bash
python scripts/07_create_sft_data.py \
    --input data/processed/corpus.jsonl \
    --output data/processed/sft_data.jsonl \
    --max-docs 200000
```

Creates template-based summarisation, Q&A, and text-continuation examples.
No model calls required — fast and deterministic.

### Step 8 — Generate SFT data (LLM-based, optional)

Generates higher-quality Q&A pairs using the CPT-merged model via vLLM.
Run this **after** CPT is complete and the adapter has been merged.

```bash
# Activate the vLLM venv (separate from training venv — see Installation)
source .venv-vllm/bin/activate

python scripts/08_generate_qa_llm.py \
    --model output/cpt/merged \
    --input data/processed/corpus.jsonl \
    --output data/processed/sft_qa_llm.jsonl \
    --qa-per-doc 3 \
    --batch-size 64
```

The CPT model is used (not the base model) so that generated questions reflect
the newly learned knowledge. Output is in the same Alpaca format as step 7 and
can be combined with or used instead of the template-based data for SFT.

## Training

### Phase 1: Continued Pretraining (CPT)

```bash
source .env          # loads HF_TOKEN, WANDB_* etc.
bash train_cpt.sh
```

Training checkpoints are saved to `output/cpt/`.
See [`configs/cpt_config.yaml`](configs/cpt_config.yaml) for all hyperparameters.

Key settings (DGX Spark / 128 GB):

| Parameter | Value | Notes |
|---|---|---|
| LoRA rank | 128 | Higher rank → more capacity for new knowledge |
| Sequence length | 8192 | Increase to 16384 if needed |
| Micro batch size | 4 | Effective batch = 32 seqs/step (128 GB headroom allows doubling) |
| Gradient checkpointing | off | Disabled: 30–40 % compute saved; VRAM budget allows it |
| Learning rate | 1e-4 | Standard for CPT |
| Precision | BF16 | No quantisation on 128 GB |
| Epochs | 1 | One pass is typically sufficient |

### Phase 2: Supervised Fine-Tuning (SFT)

After CPT completes, update `configs/sft_config.yaml` to point to the best
CPT checkpoint:

```yaml
# configs/sft_config.yaml — uncomment this line:
lora_weights: ./output/cpt/checkpoint-XXXX
```

Then run:

```bash
bash train_sft.sh
```

Output: `output/sft/`.

### Merge and export

```bash
# Merge LoRA weights into the base model for standalone deployment
accelerate launch -m axolotl.cli.merge_lora configs/sft_config.yaml \
    --lora-model-dir output/sft
```

## Project structure

```
knowledge-lora/
├── scripts/
│   ├── smoke_test.py             # Quick stack check (torch+CUDA, flash-attn, LoRA)
│   ├── 01_download_wiki.py       # Download Wikipedia dumps
│   ├── 02_extract_wiki.py        # Parse XML → JSONL
│   ├── 03_extract_pdfs.py        # PDF → JSONL
│   ├── 04_extract_markdown.py    # MD/RST/TXT → JSONL
│   ├── 05_clean_deduplicate.py   # SHA-256 + MinHash LSH dedup
│   ├── 06_tokenize.py            # Tokenise + pack → Arrow dataset
│   ├── 07_create_sft_data.py     # Template-based SFT data (no model needed)
│   └── 08_generate_qa_llm.py     # LLM-based Q&A via vLLM (run after CPT)
├── configs/
│   ├── cpt_config.yaml           # Axolotl CPT config
│   └── sft_config.yaml           # Axolotl SFT config
├── data/                         # Git-ignored; created at runtime
│   ├── raw/
│   ├── processed/
│   └── tokenized/
├── output/                       # Git-ignored; training checkpoints
├── .github/workflows/ci.yml      # Lint + type-check CI
├── .env.example                  # Credential template
├── pyproject.toml                # ruff + mypy config
├── requirements.txt
├── install.sh                    # Staged installer for CUDA 13.0 / GB10
├── install_vllm.sh               # Separate venv installer for vLLM inference
├── train_cpt.sh
├── train_sft.sh
└── LICENSE                       # Apache 2.0
```

## Adding more languages

The pipeline is language-agnostic. To add English Wikipedia:

```bash
python scripts/01_download_wiki.py --lang en
python scripts/02_extract_wiki.py --dump data/raw/enwiki-latest-pages-articles.xml.bz2 --lang en
```

Then include `data/processed/wikipedia_en.jsonl` in the `--input-files` list
for step 5. Earlier files in the list take priority during deduplication.

## Security notes

- **HF tokens** are read from the `HF_TOKEN` environment variable. Never pass
  tokens as CLI arguments (they appear in process listings and shell history).
- **Downloaded dumps** are verified against Wikimedia's published MD5 checksums
  before being used.
- **File paths** stored in JSONL output are always relative to the input directory
  to avoid leaking host filesystem layout.
- **Partial downloads** are written to a `.tmp` file and renamed atomically only
  on success. A crashed download will not leave a corrupt file at the canonical path.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE).

The base model (Ministral-3-14B-Base-2512) is also licensed under Apache 2.0.
Wikipedia content is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Ensure your custom PDFs and Markdown files are licensed for training use.
