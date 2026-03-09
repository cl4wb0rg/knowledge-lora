#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Launch CPT training with Axolotl
# Run this on the DGX Spark after the data pipeline is complete.
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

# Activate training venv (nohup/cron starts with a bare env)
source "$REPO/.venv/bin/activate"

# Load HF_TOKEN, optional WANDB_* etc.
set -a; source .env; set +a

export AXOLOTL_DO_NOT_TRACK=1

echo "=== Starting CPT training ==="
echo "Config : configs/cpt_config.yaml"
echo "Output : output/cpt"
echo ""

# Auto-resume: find latest checkpoint and pass it as a path string.
# axolotl's pydantic schema requires a string path, not a boolean.
LATEST_CKPT=$(ls -d output/cpt/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from checkpoint: $LATEST_CKPT"
    accelerate launch -m axolotl.cli.train configs/cpt_config.yaml \
        --resume_from_checkpoint "$LATEST_CKPT"
else
    echo "No checkpoint found — starting from scratch"
    accelerate launch -m axolotl.cli.train configs/cpt_config.yaml
fi

echo ""
echo "=== CPT training complete ==="
echo "Adapter saved to: output/cpt"
