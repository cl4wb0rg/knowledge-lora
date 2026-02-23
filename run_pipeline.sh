#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# knowledge-lora end-to-end pipeline orchestration
#
# Picks up after step 06 (tokenize) finishes, then runs:
#   wait-for-06 → commit → CPT training → commit → merge CPT LoRA →
#   step 08 QA gen → commit → update SFT config → SFT training → commit
#
# Run as: nohup bash run_pipeline.sh >> logs/pipeline_orchestration.log 2>&1 &
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO="/home/mvdb/knowledge-lora"
cd "$REPO"

# Activate training venv (nohup starts with a bare env)
source .venv/bin/activate

# Load .env (HF_TOKEN, optional WANDB_*)
set -a; source .env; set +a

LOG="$REPO/logs/pipeline_orchestration.log"
mkdir -p logs

log() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    printf '[%s] %s\n' "$ts" "$*" | tee -a "$LOG"
}

# Update a single table row in README.md via Python regex
readme_update() {
    # args: old_pattern  new_row
    python3 - "$1" "$2" <<'PYEOF'
import sys, re
pattern, replacement = sys.argv[1], sys.argv[2]
content = open('README.md').read()
new_content = re.sub(pattern, replacement, content)
if new_content == content:
    print(f"WARNING: pattern not matched: {pattern}", file=sys.stderr)
open('README.md', 'w').write(new_content)
PYEOF
}

git_commit_push() {
    local msg="$1"
    # Stage README and sft_config if changed
    git add README.md configs/sft_config.yaml 2>/dev/null || true
    if git diff --cached --quiet; then
        log "  Nothing new to commit for: $msg"
        return 0
    fi
    git commit -m "${msg}

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    git push origin master
    log "  Pushed: $msg"
}

# ─────────────────────────────────────────────────────────────────────────────
# WAIT FOR STEP 06 (tokenize)
# ─────────────────────────────────────────────────────────────────────────────
log "=== Waiting for step 06 (tokenize) to finish ==="
TOKENIZED_DIR="data/tokenized/cpt_dataset"

while true; do
    # Dataset is written atomically by HF datasets (dataset_info.json is last)
    if [ -f "${TOKENIZED_DIR}/dataset_info.json" ]; then
        log "  dataset_info.json found — tokenization complete."
        break
    fi
    log "  Still tokenizing… checking again in 2 min"
    sleep 120
done

# Count chunks
CHUNK_COUNT=$(python3 - <<'PYEOF'
from datasets import load_from_disk
ds = load_from_disk("data/tokenized/cpt_dataset")
print(len(ds))
PYEOF
)
log "Step 06 done: $CHUNK_COUNT packed sequences"

readme_update \
    $'\\| 06 tokenize \\| 🔄 running \\|[^|]*\\|[^|]*\\|' \
    "| 06 tokenize | ✅ done | \`data/tokenized/cpt_dataset/\` ($CHUNK_COUNT sequences) | Ministral tokenizer, seq_len 8192 |"

git_commit_push "data: complete step 06 tokenize ($CHUNK_COUNT packed sequences)"

# ─────────────────────────────────────────────────────────────────────────────
# CPT TRAINING
# ─────────────────────────────────────────────────────────────────────────────
log "=== Starting CPT training ==="

readme_update \
    $'\\| CPT training \\| ⏳ pending \\|[^|]*\\|[^|]*\\|' \
    "| CPT training | 🔄 running | \`output/cpt/\` | axolotl, LoRA rank 128 |"
git_commit_push "training: start CPT training (axolotl + LoRA rank 128)"

bash train_cpt.sh 2>&1 | tee logs/cpt_training.log

# Find last checkpoint (axolotl saves checkpoint-NNNN and optionally checkpoint-final)
CPT_CHECKPOINT=$(ls -d output/cpt/checkpoint-* 2>/dev/null | sort -V | tail -1 || echo "output/cpt")
log "CPT training done. Using checkpoint: $CPT_CHECKPOINT"

readme_update \
    $'\\| CPT training \\| 🔄 running \\|[^|]*\\|[^|]*\\|' \
    "| CPT training | ✅ done | \`output/cpt/\` | LoRA rank 128; $(basename "$CPT_CHECKPOINT") |"
git_commit_push "training: complete CPT training ($(basename "$CPT_CHECKPOINT"))"

# ─────────────────────────────────────────────────────────────────────────────
# MERGE CPT LoRA
# ─────────────────────────────────────────────────────────────────────────────
log "=== Merging CPT LoRA adapter ==="
MERGED_CPT="output/cpt/merged"
mkdir -p "$MERGED_CPT"

python -m axolotl.cli.merge_lora configs/cpt_config.yaml \
    --lora-model-dir "$CPT_CHECKPOINT" \
    --output-dir "$MERGED_CPT" 2>&1 | tee logs/cpt_merge.log \
    || {
        # Fallback: some axolotl versions use a different flag
        log "  First merge attempt failed; retrying without --output-dir"
        python -m axolotl.cli.merge_lora configs/cpt_config.yaml \
            --lora-model-dir "$CPT_CHECKPOINT" 2>&1 | tee -a logs/cpt_merge.log
        # Detect where the merged model landed
        MERGED_CPT=$(find output/cpt -name "config.json" -not -path "*/checkpoint-*/*" \
            | head -1 | xargs dirname || echo "$CPT_CHECKPOINT")
    }

log "CPT adapter merged → $MERGED_CPT"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 08: LLM-based QA generation (uses .venv-vllm)
# ─────────────────────────────────────────────────────────────────────────────
log "=== Starting step 08 (LLM QA generation, 20 000 docs) ==="

readme_update \
    $'\\| 08 QA generation \\| ⏳ pending \\|[^|]*\\|[^|]*\\|' \
    "| 08 QA generation | 🔄 running | \`data/processed/sft_qa_llm.jsonl\` | 20K docs × 3 QA, vLLM |"
git_commit_push "data: start step 08 LLM QA generation"

# Switch to vLLM venv
source .venv-vllm/bin/activate

python scripts/08_generate_qa_llm.py \
    --model "$MERGED_CPT" \
    --input  data/processed/corpus.jsonl \
    --output data/processed/sft_qa_llm.jsonl \
    --max-docs 20000 \
    --qa-per-doc 3 \
    --batch-size 32 \
    --max-new-tokens 512 2>&1 | tee logs/08_qa_gen.log

# Switch back to training venv
source .venv/bin/activate

QA_LINES=$(wc -l < data/processed/sft_qa_llm.jsonl 2>/dev/null || echo "0")
log "Step 08 done: $QA_LINES QA pairs"

readme_update \
    $'\\| 08 QA generation \\| 🔄 running \\|[^|]*\\|[^|]*\\|' \
    "| 08 QA generation | ✅ done | \`data/processed/sft_qa_llm.jsonl\` ($QA_LINES pairs) | 20K docs × 3 QA, vLLM |"
git_commit_push "data: complete step 08 QA generation ($QA_LINES pairs)"

# ─────────────────────────────────────────────────────────────────────────────
# UPDATE SFT CONFIG
# ─────────────────────────────────────────────────────────────────────────────
log "=== Updating configs/sft_config.yaml ==="
python3 - "$CPT_CHECKPOINT" <<'PYEOF'
import sys, re

checkpoint = sys.argv[1]
path = "configs/sft_config.yaml"
content = open(path).read()

# Uncomment lora_weights and point to CPT checkpoint
content = re.sub(
    r"# lora_weights:.*",
    f"lora_weights: ./{checkpoint}",
    content,
)

# Add sft_qa_llm.jsonl as a second alpaca dataset (idempotent)
if "sft_qa_llm.jsonl" not in content:
    content = content.replace(
        "  - path: data/processed/sft_data.jsonl\n    type: alpaca",
        "  - path: data/processed/sft_data.jsonl\n    type: alpaca"
        "\n  - path: data/processed/sft_qa_llm.jsonl\n    type: alpaca",
    )

open(path, "w").write(content)
print(f"Updated {path}: lora_weights → {checkpoint}, added sft_qa_llm.jsonl")
PYEOF

git_commit_push "config: point SFT to CPT checkpoint + add QA dataset"

# ─────────────────────────────────────────────────────────────────────────────
# SFT TRAINING
# ─────────────────────────────────────────────────────────────────────────────
log "=== Starting SFT training ==="

readme_update \
    $'\\| SFT training \\| ⏳ pending \\|[^|]*\\|[^|]*\\|' \
    "| SFT training | 🔄 running | \`output/sft/\` | axolotl, LoRA rank 64 |"
git_commit_push "training: start SFT training"

bash train_sft.sh 2>&1 | tee logs/sft_training.log

log "SFT training done."

readme_update \
    $'\\| SFT training \\| 🔄 running \\|[^|]*\\|[^|]*\\|' \
    "| SFT training | ✅ done | \`output/sft/\` | LoRA rank 64 |"
git_commit_push "training: complete SFT training"

log "=== PIPELINE COMPLETE — all steps done! ==="
