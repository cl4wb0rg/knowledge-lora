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

echo "==> Step 5: xformers (binary-only — skipped on CUDA 13.0, no wheel available)"
# xformers uses removed CUDA 13.0 driver API symbols (PFN_cuGetErrorName etc.)
# and cannot be built from source. --only-binary prevents pip from ever
# attempting a source compile (which spawns many parallel cicc/nvcc processes
# and exhausts all RAM+swap, freezing the system). If no pre-built wheel
# matches the current platform, pip fails immediately and we skip gracefully.
# axolotl's flash_attention: true config uses flash-attn instead.
pip install "xformers==0.0.28.post2" \
    --only-binary xformers \
    --extra-index-url "${TORCH_INDEX}" \
    --quiet \
    || echo "    xformers skipped — flash-attn will be used instead (see configs)"

# NOTE: vLLM is NOT installed here.
# vLLM 0.15.x requires torch==2.9.1 while axolotl HEAD requires torch==2.10.0.
# Installing both in the same venv causes torch to be downgraded, which then
# breaks flash-attn (compiled C extensions become ABI-incompatible).
# Use a separate venv for vLLM inference — see install_vllm.sh.

echo "==> Step 6: flash-attn (built from source against CUDA 13.0 — takes ~20 min)"
pip install wheel --quiet  # flash-attn setup.py requires 'wheel' in the venv

# Skip rebuild if flash-attn already imports correctly (saves ~20 min on re-runs).
if python -c "import flash_attn; import torch; assert torch.version.cuda is not None" 2>/dev/null; then
    echo "    flash-attn already installed and torch CUDA OK — skipping rebuild"
else
    # --no-binary forces source build; the pre-built wheel links libcudart.so.12
    # (CUDA 12) which is absent on this system (CUDA 13.0).
    # MAX_JOBS=1 limits to one parallel nvcc process to stay below 80% CPU/RAM.
    # nice -n 10 reduces build priority so the system stays responsive.
    # --extra-index-url: prevents pip from pulling CPU torch from plain PyPI
    #   when re-resolving flash-attn's deps.
    # --- Pre-build safeguards (require sudo; silently skipped if unavailable) ---
    if sudo -n true 2>/dev/null; then
        echo "    [safeguard] clearing filesystem cache..."
        sudo -n sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'
        # Cap GPU power to 80% of max to prevent overcurrent spikes during nvcc compilation.
        # Stores the original limit so it can be restored after the build.
        _GPU_ORIG_W=$(nvidia-smi --query-gpu=power.limit \
            --format=csv,noheader,nounits 2>/dev/null | awk '{printf "%d", $1}')
        _GPU_CAP_W=$(nvidia-smi --query-gpu=power.max_limit \
            --format=csv,noheader,nounits 2>/dev/null | awk '{printf "%d", $1 * 0.8}')
        if [ -n "$_GPU_CAP_W" ] && [ "$_GPU_CAP_W" -gt 0 ]; then
            sudo -n nvidia-smi -pl "$_GPU_CAP_W" \
                && echo "    [safeguard] GPU power capped to ${_GPU_CAP_W}W (was ${_GPU_ORIG_W}W)"
        fi
    else
        echo "    [safeguard] sudo unavailable — cache clear and GPU power cap skipped"
    fi

    echo "    starting build (output visible so terminal stays 'alive')..."
    ( while true; do sleep 60; echo "    [flash-attn still building...]"; done ) &
    _HEARTBEAT=$!
    # Mark this shell as high-priority for OOM kill: if RAM fills up, the kernel
    # kills this build process first (graceful failure) instead of freezing the system.
    echo 500 > /proc/self/oom_score_adj 2>/dev/null || true
    MAX_JOBS=1 nice -n 10 pip install flash-attn \
        --no-build-isolation \
        --no-binary flash-attn \
        --force-reinstall \
        --no-cache-dir \
        --extra-index-url "${TORCH_INDEX}"
    kill "$_HEARTBEAT" 2>/dev/null; wait "$_HEARTBEAT" 2>/dev/null || true

    # Restore GPU power limit after build
    if [ -n "${_GPU_ORIG_W:-}" ] && [ "${_GPU_ORIG_W:-0}" -gt 0 ] && sudo -n true 2>/dev/null; then
        sudo -n nvidia-smi -pl "$_GPU_ORIG_W" 2>/dev/null \
            && echo "    [safeguard] GPU power limit restored to ${_GPU_ORIG_W}W"
    fi
fi

echo "==> Step 6b: re-pin torch+cu130 and fsspec (flash-attn dep resolver may replace them)"
# --force-reinstall above can still pull CPU torch or a newer fsspec from PyPI.
# Re-installing here guarantees the CUDA build and a datasets-compatible fsspec.
pip install "torch==2.10.0+cu130" --index-url "${TORCH_INDEX}" --quiet
pip install "fsspec>=2023.1.0,<=2025.10.0" --quiet

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
