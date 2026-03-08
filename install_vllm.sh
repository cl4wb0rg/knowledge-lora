#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Separate venv for vLLM inference (08_generate_qa_llm.py)
#
# WHY a separate venv and not the training venv?
#   axolotl HEAD requires torch==2.10.0+cu130
#   vLLM 0.16.x requires torch==2.9.1
#   → incompatible in the same venv; installing vLLM downgrades torch and
#     breaks flash-attn (ABI mismatch on compiled C extensions).
#
# WHY not Docker (e.g. nvcr.io/nvidia/vllm:26.02-py3)?
#   The Docker approach was tried first.  It requires a ~30 GB image pull,
#   needs --gpus all + volume mounts, and does not share the host CUDA 13
#   libraries cleanly on the GB10.  The patched-venv approach is lighter,
#   starts faster after the first install, and gives direct access to the
#   local merged model weights without any bind-mount configuration.
#
# WHY binary patches (CUDA 13.0 only)?
#   vLLM 0.16.0 wheels are built against libcudart.so.12 (CUDA 12.x).
#   CUDA 13.0 ships libcudart.so.13.  The dynamic linker refuses to load
#   the vLLM .so files because:
#     1. DT_NEEDED entry in .dynstr still says "libcudart.so.12"
#     2. vna_hash in .gnu.version_r still holds elf_hash("libcudart.so.12")
#        and glibc checks vna_hash == vd_hash for version matching.
#   Both fields must be patched.  scripts/patch_vllm_verneed_hash.py handles
#   step 2 (step 1 is an inline sed-style Python below).
#   After both patches "from vllm import LLM" works on CUDA 13.0.
#
# Usage:
#   bash install_vllm.sh              # creates .venv-vllm/ and patches binaries
#   source .venv-vllm/bin/activate
#   python scripts/08_generate_qa_llm.py --model output/cpt/merged ...
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

VENV=".venv-vllm"
CUDA_TAG="cu130"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
VLLM_VERSION="0.16.0"   # pinned — binary patches are version-specific

echo "==> Creating venv: ${VENV}"
python -m venv "${VENV}"
# shellcheck disable=SC1090
source "${VENV}/bin/activate"

echo "==> PyTorch for vLLM (torch==2.9.1+cu130)"
pip install "torch==2.9.1+cu130" --index-url "${TORCH_INDEX}" --quiet

echo "==> vLLM ${VLLM_VERSION}"
pip install "vllm==${VLLM_VERSION}" \
    --extra-index-url "${TORCH_INDEX}" \
    --timeout 120 \
    --quiet

echo "==> Minimal extras for 08_generate_qa_llm.py"
pip install sentencepiece tqdm pyelftools --quiet

# ─────────────────────────────────────────────────────────────────────────────
# CUDA 13.0 binary patches
# Skip if not on CUDA 13.0 (libcudart.so.13 present means we're on CUDA 13).
# ─────────────────────────────────────────────────────────────────────────────
if ldconfig -p 2>/dev/null | grep -q "libcudart.so.13"; then
    echo "==> CUDA 13.0 detected — applying vLLM binary patches"

    # Step 1: patch DT_NEEDED string in .dynstr: libcudart.so.12 -> libcudart.so.13
    echo "    Step 1: patch DT_NEEDED strings"
    python3 - "${VENV}" <<'PYEOF'
import sys
from pathlib import Path

venv = Path(sys.argv[1])
OLD = b'libcudart.so.12\x00'
NEW = b'libcudart.so.13\x00'
assert len(OLD) == len(NEW)

patched = 0
for p in venv.rglob('*.so'):
    try:
        data = p.read_bytes()
    except PermissionError:
        continue
    if OLD in data:
        p.write_bytes(data.replace(OLD, NEW))
        print(f"    DT_NEEDED patched: {p.name}")
        patched += 1

print(f"    Step 1 done: {patched} file(s) patched")
PYEOF

    # Step 2: patch vna_hash in .gnu.version_r sections
    # elf_hash("libcudart.so.12")=0x0ff3db22, elf_hash("libcudart.so.13")=0x0ff3db23
    echo "    Step 2: patch vna_hash in .gnu.version_r"
    python3 scripts/patch_vllm_verneed_hash.py

    echo "    Binary patches done — vLLM ready for CUDA 13.0"
else
    echo "==> CUDA 13.0 not detected — skipping binary patches"
fi

echo ""
echo "Done. Activate with:"
echo "  source ${VENV}/bin/activate"
echo ""
echo "Then run:"
echo "  python scripts/08_generate_qa_llm.py \\"
echo "      --model output/cpt/merged \\"
echo "      --input data/processed/corpus.jsonl \\"
echo "      --output data/processed/sft_qa_llm.jsonl"
