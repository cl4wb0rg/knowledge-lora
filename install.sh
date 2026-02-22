#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap installation for knowledge-lora on DGX Spark (GB10 / CUDA 13.0)
#
# Why not plain `pip install -r requirements.txt`?
#   1. xformers requires torch at build-time — pip doesn't guarantee order.
#   2. xformers has no pre-built cu130 wheel — we skip it; flash-attn covers it.
#   3. Pre-pinning axolotl's transitive deps (transformers, peft, accelerate)
#      before axolotl itself causes ResolutionImpossible conflicts. We let
#      axolotl resolve its own dependency tree first, then add extras.
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

echo "==> Step 2: Axolotl + DeepSpeed"
# Install axolotl BEFORE pinning its transitive deps (transformers, peft, etc.).
# --no-build-isolation makes the already-installed torch visible to xformers
# if it attempts a source build; we also skip xformers explicitly via constraint.
pip install "axolotl[deepspeed]>=0.6.0,<1.0" \
    --extra-index-url "${TORCH_INDEX}" \
    --no-build-isolation \
    --constraint <(echo "xformers==0.0.0" 2>/dev/null || true) \
    --quiet \
    || pip install "axolotl[deepspeed]>=0.6.0,<1.0" \
        --extra-index-url "${TORCH_INDEX}" \
        --no-build-isolation \
        --quiet

echo "==> Step 3: Data pipeline"
pip install \
    wikiextractor \
    "pymupdf>=1.24.0,<2.0" \
    datasketch \
    sentencepiece protobuf tqdm \
    --quiet

echo "==> Step 4: bitsandbytes (optional, for 8-bit experiments)"
pip install "bitsandbytes>=0.44.0" --quiet \
    || echo "    bitsandbytes skipped (non-critical)"

echo "==> Step 5: xformers (best-effort — skipped if no cu130 wheel available)"
pip install "xformers==0.0.29.post2" \
    --extra-index-url "${TORCH_INDEX}" \
    --no-build-isolation \
    --quiet \
    || echo "    xformers skipped — flash-attn will be used instead (see configs)"

echo "==> Step 6: vLLM (for LLM-based Q&A generation — 08_generate_qa_llm.py)"
pip install "vllm>=0.6.0,<1.0" \
    --extra-index-url "${TORCH_INDEX}" \
    --quiet \
    || echo "    vLLM skipped — install manually after torch is confirmed working"

echo "==> Step 7: flash-attn (built from source — takes a few minutes)"
pip install flash-attn --no-build-isolation --quiet

echo ""
echo "Done. Verify with:"
echo "  python -c \"import torch; print(torch.__version__, torch.cuda.get_device_name(0))\""
