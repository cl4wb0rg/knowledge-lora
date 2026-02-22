#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap installation for knowledge-lora on DGX Spark (GB10 / CUDA 13.0)
#
# Why not plain `pip install -r requirements.txt`?
#   1. xformers requires torch at build-time — pip doesn't guarantee order.
#   2. xformers has no pre-built cu130 wheel — we skip it; flash-attn covers it.
#   3. axolotl GitHub HEAD pins exact versions (torch, transformers, hf-hub) —
#      we install those exact versions after axolotl to avoid conflicts.
#
# Usage:
#   bash install.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CUDA_TAG="cu130"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
TORCH_NIGHTLY="https://download.pytorch.org/whl/nightly/${CUDA_TAG}"

echo "==> Step 1: PyTorch (CUDA ${CUDA_TAG})"
# axolotl HEAD currently requires torch==2.10.0+cu130; try that first,
# then fall back to latest stable, then nightly.
if pip install "torch==2.10.0+cu130" --index-url "${TORCH_INDEX}" --quiet 2>/dev/null; then
    echo "    torch 2.10.0+cu130 installed"
elif pip install "torch>=2.9.0,<3.0" --index-url "${TORCH_INDEX}" --quiet; then
    echo "    torch $(python -c 'import torch; print(torch.__version__)') installed (latest stable)"
else
    echo "    stable cu130 wheels not found — falling back to nightly"
    pip install --pre "torch" --index-url "${TORCH_NIGHTLY}" --quiet
fi
python -c "import torch; print('    torch', torch.__version__)"

echo "==> Step 2: Axolotl (GitHub HEAD, no DeepSpeed)"
# deepspeed-kernels has no Python 3.12 wheel — and single-GPU LoRA on 128 GB
# doesn't need DeepSpeed/ZeRO anyway. Install core axolotl only.
pip install \
    "axolotl @ git+https://github.com/axolotl-ai-cloud/axolotl.git" \
    --extra-index-url "${TORCH_INDEX}" \
    --no-build-isolation \
    --quiet

echo "==> Step 2b: Align packages with axolotl HEAD requirements"
# axolotl 0.15.0.dev0 pins transformers and huggingface-hub to specific
# versions. Upgrading here prevents runtime import errors.
# numpy is required by flash-attn C extensions at import time.
pip install \
    "transformers==5.2.0" \
    "huggingface-hub>=1.1.7" \
    numpy \
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

echo "==> Step 5: xformers (best-effort — skipped if CUDA 13.0 build fails)"
# xformers uses removed CUDA 13.0 driver API symbols (PFN_cuGetErrorName etc.)
# and cannot be built from source. axolotl's flash_attention: true config
# uses flash-attn instead, which is fully functional.
pip install "xformers==0.0.28.post2" \
    --extra-index-url "${TORCH_INDEX}" \
    --no-build-isolation \
    --quiet \
    || echo "    xformers skipped — flash-attn will be used instead (see configs)"

# NOTE: vLLM is NOT installed here.
# vLLM 0.15.x requires torch==2.9.1 while axolotl HEAD requires torch==2.10.0.
# Installing both in the same venv causes torch to be downgraded, which then
# breaks flash-attn (compiled C extensions become ABI-incompatible).
# Use a separate venv for vLLM inference — see install_vllm.sh.

echo "==> Step 6: flash-attn (built from source — takes a few minutes)"
pip install wheel --quiet  # flash-attn setup.py requires 'wheel' in the venv
pip install flash-attn --no-build-isolation

echo ""
echo "==> Dependency check"
python - <<'EOF'
import importlib, sys
ok = True
for pkg, imp in [
    ("torch", "torch"), ("transformers", "transformers"),
    ("axolotl", "axolotl"), ("peft", "peft"),
    ("accelerate", "accelerate"), ("flash_attn", "flash_attn"),
    ("datasets", "datasets"),
]:
    try:
        m = importlib.import_module(imp)
        ver = getattr(m, "__version__", "?")
        print(f"    OK  {pkg}=={ver}")
    except ImportError:
        print(f"    MISSING  {pkg}", file=sys.stderr)
        ok = False
sys.exit(0 if ok else 1)
EOF

echo ""
echo "Done. Verify GPU access with:"
echo "  python -c \"import torch; print(torch.__version__, torch.cuda.get_device_name(0))\""
