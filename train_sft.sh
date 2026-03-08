#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Launch SFT training with Axolotl
# Run this AFTER CPT is complete and lora_weights is set in sft_config.yaml
# (run_pipeline.sh does this automatically after merging the CPT LoRA).
#
# Known issues fixed vs. the naive invocation:
#   - AXOLOTL_DO_NOT_TRACK=1 : whitelist.yaml is missing in GitHub HEAD installs
#   - source .env            : loads HF_TOKEN so gated model download works
#   - auto-resume            : passes --resume_from_checkpoint if a checkpoint
#                              exists in output/sft/ (axolotl pydantic requires
#                              a string path, not resume_from_checkpoint: true)
#   - REPO + cd              : works correctly whether called directly or via
#                              run_pipeline.sh (which already cd'd to REPO)
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

# Load HF_TOKEN, optional WANDB_* etc.
set -a; source .env; set +a

# axolotl GitHub HEAD installs lack whitelist.yaml — disable telemetry to avoid crash
export AXOLOTL_DO_NOT_TRACK=1

echo "=== Starting SFT training ==="
echo "Config : configs/sft_config.yaml"
echo "Output : output/sft"
echo ""

# Auto-resume: if a checkpoint exists in output/sft/, continue from it.
# axolotl's pydantic schema requires a string path, not a boolean.
LATEST_CKPT=$(ls -d output/sft/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from checkpoint: $LATEST_CKPT"
    accelerate launch -m axolotl.cli.train configs/sft_config.yaml \
        --resume_from_checkpoint "$LATEST_CKPT"
else
    echo "No checkpoint found — starting from scratch"
    accelerate launch -m axolotl.cli.train configs/sft_config.yaml
fi

echo ""
echo "=== SFT training complete ==="
echo "Adapter saved to: output/sft"
echo ""
echo "To merge the SFT LoRA into the base model:"
echo "  accelerate launch -m axolotl.cli.merge_lora configs/sft_config.yaml \\"
echo "      --lora-model-dir output/sft"
