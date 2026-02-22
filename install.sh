#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap installation for knowledge-lora on DGX Spark (GB10 / CUDA 13.0)
#
# Why not plain `pip install -r requirements.txt`?
#   xformers requires torch at build-time; pip doesn't guarantee install order.
#   xformers also has no pre-built wheels for CUDA 13.0 yet — we skip it and
#   rely on flash-attn (configured in axolotl configs via flash_attention: true).
#
# Usage:
#   bash install.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CUDA_TAG="cu130"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"

echo "==> Step 1: PyTorch (CUDA ${CUDA_TAG})"
if pip install "torch>=2.3.0,<3.0" --index-url "${TORCH_INDEX}" --quiet; then
    echo "    torch installed from ${TORCH_INDEX}"
else
    echo "    cu130 wheels not found — falling back to PyTorch nightly"
    pip install --pre "torch" \
        --index-url "https://download.pytorch.org/whl/nightly/${CUDA_TAG}" \
        --quiet
fi

echo "==> Step 2: Core dependencies (excluding xformers)"
pip install \
    "transformers>=4.47.0,<5.0" \
    "peft>=0.14.0,<1.0" \
    "accelerate>=0.34.0,<2.0" \
    "datasets>=3.0.0,<4.0" \
    "huggingface_hub>=0.26.0,<2.0" \
    sentencepiece protobuf tqdm \
    --extra-index-url "${TORCH_INDEX}" \
    --quiet

echo "==> Step 3: Data pipeline"
pip install \
    wikiextractor \
    "pymupdf>=1.24.0,<2.0" \
    datasketch \
    --quiet

echo "==> Step 4: bitsandbytes (optional, for 8-bit experiments)"
pip install "bitsandbytes>=0.44.0" --quiet || echo "    bitsandbytes skipped (non-critical)"

echo "==> Step 5: Axolotl + DeepSpeed"
# Install without triggering a xformers source build.
# --no-build-isolation ensures the already-installed torch is visible.
pip install "axolotl[deepspeed]>=0.6.0,<1.0" \
    --extra-index-url "${TORCH_INDEX}" \
    --no-build-isolation \
    --quiet

echo "==> Step 6: xformers (best-effort — skipped if no cu130 wheel available)"
pip install "xformers==0.0.29.post2" \
    --extra-index-url "${TORCH_INDEX}" \
    --no-build-isolation \
    --quiet \
    || echo "    xformers skipped — flash-attn will be used instead (see configs)"

echo "==> Step 7: vLLM (for LLM-based Q&A generation — 08_generate_qa_llm.py)"
pip install "vllm>=0.6.0,<1.0" \
    --extra-index-url "${TORCH_INDEX}" \
    --quiet \
    || echo "    vLLM skipped — install manually after torch is confirmed working"

echo "==> Step 8: flash-attn (built from source — takes a few minutes)"
pip install flash-attn --no-build-isolation --quiet

echo ""
echo "Done. Verify with:"
echo "  python -c \"import torch; print(torch.__version__, torch.cuda.get_device_name(0))\""
