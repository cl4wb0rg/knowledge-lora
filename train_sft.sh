#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Launch SFT training with Axolotl
# Run this AFTER CPT is complete and you have uncommented lora_weights
# in configs/sft_config.yaml.
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# Verify CPT adapter exists
CPT_DIR="output/cpt"
if [ ! -d "$CPT_DIR" ]; then
    echo "ERROR: CPT adapter not found at $CPT_DIR"
    echo "Run train_cpt.sh first, then update lora_weights in configs/sft_config.yaml"
    exit 1
fi

# Optional: uncomment to set HF token
# export HF_TOKEN="hf_..."

# Optional: Weights & Biases
# export WANDB_PROJECT="ministral-14b-lora"
# export WANDB_RUN_NAME="sft-de-wiki"

echo "=== Starting SFT training ==="
echo "Config : configs/sft_config.yaml"
echo "Output : output/sft"
echo ""

accelerate launch -m axolotl.cli.train configs/sft_config.yaml

echo ""
echo "=== SFT training complete ==="
echo "Final adapter saved to: output/sft"
echo ""
echo "To merge and push to Hub:"
echo "  axolotl.cli.merge_lora configs/sft_config.yaml --lora-model-dir output/sft"
