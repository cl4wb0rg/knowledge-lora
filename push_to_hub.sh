#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# push_to_hub.sh — Upload LoRA adapters and datasets to HuggingFace Hub
#
# Pushes any combination of:
#   CPT LoRA adapter   output/cpt/checkpoint-2400/      → {user}/ministral-14b-de-cpt-lora
#   SFT LoRA adapter   output/sft/ or checkpoint-*      → {user}/ministral-14b-de-sft-lora
#   SFT datasets       data/processed/sft_*.jsonl       → {user}/ministral-14b-de-dataset
#
# The Hub username is resolved from HF_TOKEN automatically — never hardcoded.
#
# Usage:
#   bash push_to_hub.sh               # push everything that exists (private)
#   bash push_to_hub.sh --cpt         # CPT adapter only
#   bash push_to_hub.sh --sft         # SFT adapter only
#   bash push_to_hub.sh --data        # datasets only
#   bash push_to_hub.sh --public      # make repos public (default: private)
#   bash push_to_hub.sh --prefix myorg  # repo prefix instead of your username
#
# Security notes:
#   - HF_TOKEN is read from .env (gitignored) — never hardcoded or logged
#   - xtrace (set -x) is intentionally not enabled — it would log the token
#   - Repos are private by default; pass --public to override
#   - Optimizer states (optimizer.pt, 3.7 GB) are excluded — inference only
# ═══════════════════════════════════════════════════════════════════════════════

# SECURITY: never enable xtrace in this script
set +x
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

# ── Load credentials ──────────────────────────────────────────────────────────
# .env is gitignored; contains HF_TOKEN and optionally WANDB_* etc.
# set -a exports all sourced variables to child processes (Python, huggingface-cli)
set -a; source .env; set +a

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set."
    echo "  Add it to .env:  HF_TOKEN=hf_..."
    echo "  Create tokens at: https://huggingface.co/settings/tokens"
    exit 1
fi

# ── Argument parsing ──────────────────────────────────────────────────────────
DO_CPT=false; DO_SFT=false; DO_DATA=false
PRIVATE=true
PREFIX=""   # empty = use HF username

[[ $# -eq 0 ]] && DO_CPT=true && DO_SFT=true && DO_DATA=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpt)    DO_CPT=true ;;
        --sft)    DO_SFT=true ;;
        --data)   DO_DATA=true ;;
        --public) PRIVATE=false ;;
        --prefix)
            shift
            PREFIX="${1:?--prefix requires a value}"
            ;;
        *)
            echo "Unknown flag: $1"
            echo "Usage: bash push_to_hub.sh [--cpt] [--sft] [--data] [--public] [--prefix ORG]"
            exit 1
            ;;
    esac
    shift
done

# ── Activate training venv (has huggingface_hub + datasets installed) ─────────
source "$REPO/.venv/bin/activate"

# ── Resolve Hub username ───────────────────────────────────────────────────────
# whoami() uses HF_TOKEN from the environment — token never echoed
HF_USERNAME=$(python3 -c "
from huggingface_hub import whoami
info = whoami()
print(info['name'])
")

OWNER="${PREFIX:-$HF_USERNAME}"
VIS="$( [ "$PRIVATE" = true ] && echo private || echo PUBLIC )"
[ "$PRIVATE" = false ] && echo "WARNING: repositories will be PUBLIC on HuggingFace" && echo ""

echo "Hub user  : $HF_USERNAME"
echo "Repo owner: $OWNER"
echo "Visibility: $VIS"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Helper: upload a local directory as a model repo
# Excludes large training-only files that are not needed for inference.
# ─────────────────────────────────────────────────────────────────────────────
push_adapter() {
    local local_dir="$1"
    local repo_id="$2"
    local description="$3"

    if [ ! -f "$local_dir/adapter_config.json" ]; then
        echo "SKIP $repo_id: adapter_config.json not found in $local_dir"
        return 0
    fi

    echo "==> $description"
    echo "    $local_dir → $OWNER/$repo_id"

    python3 - "$local_dir" "$OWNER/$repo_id" "$PRIVATE" "$description" <<'PYEOF'
import sys, os
from pathlib import Path
from huggingface_hub import HfApi, ModelCard, ModelCardData

local_dir, repo_id, private_str, description = sys.argv[1:]
private = (private_str == "true")
local_dir = Path(local_dir)

api = HfApi()

# Create repo if it doesn't exist yet
api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

# Build a proper model card (replaces the blank axolotl template)
card_data = ModelCardData(
    base_model="mistralai/Ministral-3-14B-Base-2512",
    library_name="peft",
    pipeline_tag="text-generation",
    tags=["lora", "axolotl", "german", "wikipedia", "knowledge-lora"],
    language=["de", "en"],
    license="apache-2.0",
)
card_content = f"""---
{card_data.to_yaml()}
---

# {repo_id.split("/")[-1]}

{description}

**Base model**: [mistralai/Ministral-3-14B-Base-2512](https://huggingface.co/mistralai/Ministral-3-14B-Base-2512)
**Training code**: [knowledge-lora](https://github.com/cl4wb0rg/knowledge-lora)

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "mistralai/Ministral-3-14B-Base-2512",
    torch_dtype="bfloat16",
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```

## Training details

| Parameter | Value |
|---|---|
| Architecture | Ministral-3-14B-Base-2512 (13.8 B params) |
| Adapter | LoRA |
| Precision | BF16 |
| Training data | German Wikipedia + custom documents |
| Training framework | [axolotl](https://github.com/axolotl-ai-cloud/axolotl) |
"""

ModelCard(card_content).save(local_dir / "README.md")

# Upload — skip optimizer states and training-only artifacts
api.upload_folder(
    folder_path=str(local_dir),
    repo_id=repo_id,
    repo_type="model",
    ignore_patterns=[
        "optimizer.pt",
        "optimizer.bin",
        "rng_state*.pth",
        "training_args.bin",
        "tokens_state*",
        "*.tmp",
        "__pycache__",
        ".cache",
    ],
    commit_message=f"Upload {description}",
)

size_gb = sum(
    f.stat().st_size for f in local_dir.rglob("*") if f.is_file()
    and f.name not in ("optimizer.pt", "optimizer.bin")
) / 1e9
print(f"    Uploaded ~{size_gb:.1f} GB → https://huggingface.co/{repo_id}")
PYEOF
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper: push a JSONL file as one split of a dataset repo
# ─────────────────────────────────────────────────────────────────────────────
push_dataset_split() {
    local jsonl_path="$1"
    local repo_id="$2"
    local split="$3"
    local description="$4"

    if [ ! -f "$jsonl_path" ]; then
        echo "SKIP dataset split '$split': $jsonl_path not found"
        return 0
    fi

    echo "==> Dataset split: $split"
    echo "    $jsonl_path → $OWNER/$repo_id (split=$split)"

    python3 - "$jsonl_path" "$OWNER/$repo_id" "$split" "$PRIVATE" "$description" <<'PYEOF'
import sys
from datasets import load_dataset

path, repo_id, split, private_str, description = sys.argv[1:]
private = (private_str == "true")

ds = load_dataset("json", data_files=path, split="train")
ds.push_to_hub(
    repo_id,
    split=split,
    private=private,
    commit_message=f"Upload {description} ({len(ds)} rows)",
)
print(f"    {len(ds):,} rows → https://huggingface.co/datasets/{repo_id}")
PYEOF
}

# ─────────────────────────────────────────────────────────────────────────────
# CPT LoRA adapter
# ─────────────────────────────────────────────────────────────────────────────
if [ "$DO_CPT" = true ]; then
    # Use the final checkpoint (highest step number)
    CPT_CKPT=$(ls -d "$REPO"/output/cpt/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
    if [ -z "$CPT_CKPT" ]; then
        echo "SKIP CPT adapter: no checkpoints found in output/cpt/"
    else
        push_adapter \
            "$CPT_CKPT" \
            "ministral-14b-de-cpt-lora" \
            "CPT LoRA adapter — Ministral-3-14B continued pretraining on German Wikipedia"
    fi
    echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
# SFT LoRA adapter
# Best checkpoint first (output/sft/ root, saved by load_best_model_at_end),
# fallback to latest checkpoint-* subdirectory.
# ─────────────────────────────────────────────────────────────────────────────
if [ "$DO_SFT" = true ]; then
    if [ -f "$REPO/output/sft/adapter_config.json" ]; then
        SFT_DIR="$REPO/output/sft"
    else
        SFT_DIR=$(ls -d "$REPO"/output/sft/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
    fi

    if [ -z "${SFT_DIR:-}" ]; then
        echo "SKIP SFT adapter: no adapter found in output/sft/"
    else
        push_adapter \
            "$SFT_DIR" \
            "ministral-14b-de-sft-lora" \
            "SFT LoRA adapter — instruction fine-tuning on German Wikipedia Q&A"
    fi
    echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
# Datasets — both splits go to the same dataset repo
# split "template": template-based SFT data from step 07 (no model needed)
# split "llm_qa":   LLM-generated Q&A from step 08 (requires CPT model)
# ─────────────────────────────────────────────────────────────────────────────
if [ "$DO_DATA" = true ]; then
    DATASET_REPO="ministral-14b-de-dataset"

    push_dataset_split \
        "$REPO/data/processed/sft_data.jsonl" \
        "$DATASET_REPO" \
        "template" \
        "template-based SFT data (step 07)"

    push_dataset_split \
        "$REPO/data/processed/sft_qa_llm.jsonl" \
        "$DATASET_REPO" \
        "llm_qa" \
        "LLM-generated Q&A pairs (step 08)"
    echo ""
fi

echo "=== push_to_hub.sh done ==="
