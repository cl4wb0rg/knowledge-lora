#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Separate venv for vLLM inference (08_generate_qa_llm.py)
#
# WHY separate?
#   axolotl HEAD requires torch==2.10.0+cu130
#   vLLM 0.15.x requires torch==2.9.1
#   → incompatible in the same venv; installing vLLM downgrades torch and
#     breaks flash-attn (ABI mismatch on compiled C extensions).
#
# Usage:
#   bash install_vllm.sh              # creates .venv-vllm/
#   source .venv-vllm/bin/activate
#   python scripts/08_generate_qa_llm.py --model output/cpt/merged ...
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

VENV=".venv-vllm"
CUDA_TAG="cu130"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"

echo "==> Creating venv: ${VENV}"
python -m venv "${VENV}"
# shellcheck disable=SC1090
source "${VENV}/bin/activate"

echo "==> PyTorch for vLLM (torch==2.9.1+cu130)"
pip install "torch==2.9.1+cu130" --index-url "${TORCH_INDEX}" --quiet

echo "==> vLLM"
pip install "vllm>=0.15.0,<1.0" \
    --extra-index-url "${TORCH_INDEX}" \
    --timeout 120 \
    --quiet

echo "==> Minimal extras for 08_generate_qa_llm.py"
pip install sentencepiece tqdm --quiet

echo ""
echo "Done. Activate with:"
echo "  source ${VENV}/bin/activate"
echo ""
echo "Then run:"
echo "  python scripts/08_generate_qa_llm.py \\"
echo "      --model output/cpt/merged \\"
echo "      --input data/processed/corpus.jsonl \\"
echo "      --output data/processed/sft_qa_llm.jsonl"
