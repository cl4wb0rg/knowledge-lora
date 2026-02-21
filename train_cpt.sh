#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Launch CPT training with Axolotl
# Run this on the DGX Spark after the data pipeline is complete.
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# Optional: set your HuggingFace token if the model is gated
# export HF_TOKEN="hf_..."

# Optional: Weights & Biases
# export WANDB_PROJECT="ministral-14b-lora"
# export WANDB_RUN_NAME="cpt-de-wiki"

echo "=== Starting CPT training ==="
echo "Config : configs/cpt_config.yaml"
echo "Output : output/cpt"
echo ""

accelerate launch -m axolotl.cli.train configs/cpt_config.yaml

echo ""
echo "=== CPT training complete ==="
echo "Adapter saved to: output/cpt"
