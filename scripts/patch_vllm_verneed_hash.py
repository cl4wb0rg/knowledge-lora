#!/usr/bin/env python3
"""
Patch the vna_hash field in .gnu.version_r sections of vllm .so files.

After binary-patching the DT_NEEDED string from 'libcudart.so.12' to
'libcudart.so.13', the vna_hash in each Vernaux entry still holds
elf_hash("libcudart.so.12"). glibc's dynamic linker checks vna_hash
against the library's vd_hash, so the version lookup fails.

This script fixes the vna_hash values for any verneed entry whose
vna_name string is 'libcudart.so.13'.
"""

import struct
from pathlib import Path

from elftools.elf.elffile import ELFFile

TARGET_VERSION = b"libcudart.so.13\x00"


def elf_hash(s: str) -> int:
    h = 0
    for c in s.encode("ascii"):
        h = (h << 4) + c
        g = h & 0xF0000000
        if g:
            h ^= g >> 24
        h &= ~g
        h &= 0xFFFFFFFF
    return h


OLD_HASH = elf_hash("libcudart.so.12")
NEW_HASH = elf_hash("libcudart.so.13")
OLD_BYTES = struct.pack("<I", OLD_HASH)
NEW_BYTES = struct.pack("<I", NEW_HASH)

print(f"Replacing vna_hash {OLD_HASH:#010x} → {NEW_HASH:#010x}")
print(f"  bytes: {OLD_BYTES.hex()} → {NEW_BYTES.hex()}")


def patch_file(path: Path) -> int:
    """Return number of vna_hash fields patched."""
    data = bytearray(path.read_bytes())

    with path.open("rb") as f:
        elf = ELFFile(f)
        section = elf.get_section_by_name(".gnu.version_r")
        if section is None:
            print(f"  {path.name}: no .gnu.version_r section — skipping")
            return 0

        sec_off = section["sh_offset"]
        sec_size = section["sh_size"]
        dynstr = elf.get_section_by_name(".dynstr")
        dynstr_data = dynstr.data()

        # Walk the Verneed linked list manually
        # Verneed: version(2) cnt(2) file(4) aux(4) next(4) = 16 bytes
        # Vernaux: hash(4) flags(2) other(2) name(4) next(4) = 16 bytes
        patched = 0
        vn_off = 0
        while vn_off < sec_size:
            _vn_version, vn_cnt, _vn_file, vn_aux, vn_next = struct.unpack_from(
                "<HHIII", section.data(), vn_off
            )
            # Walk vernaux chain
            va_off = vn_off + vn_aux
            for _ in range(vn_cnt):
                _vna_hash, _vna_flags, _vna_other, vna_name, vna_next = struct.unpack_from(
                    "<IHHII", section.data(), va_off
                )
                # Check if this vernaux refers to 'libcudart.so.13'
                name_str = dynstr_data[vna_name:].split(b"\x00")[0].decode()
                if name_str == "libcudart.so.13":
                    file_offset = sec_off + va_off  # vna_hash is first field
                    current = struct.unpack_from("<I", data, file_offset)[0]
                    print(
                        f"  {path.name}: vernaux '{name_str}' at file offset "
                        f"0x{file_offset:x}, current hash 0x{current:08x}"
                    )
                    if current == OLD_HASH:
                        data[file_offset : file_offset + 4] = NEW_BYTES
                        patched += 1
                        print(f"    → patched to 0x{NEW_HASH:08x}")
                    elif current == NEW_HASH:
                        print("    → already correct, skipping")
                    else:
                        print(f"    → unexpected hash 0x{current:08x}, skipping")
                if vna_next == 0:
                    break
                va_off += vna_next

            if vn_next == 0:
                break
            vn_off += vn_next

    if patched > 0:
        path.write_bytes(bytes(data))
    return patched


SO_FILES = [
    ".venv-vllm/lib/python3.12/site-packages/vllm/_C.abi3.so",
    ".venv-vllm/lib/python3.12/site-packages/vllm/_flashmla_C.abi3.so",
    ".venv-vllm/lib/python3.12/site-packages/vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so",
    ".venv-vllm/lib/python3.12/site-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so",
    ".venv-vllm/lib/python3.12/site-packages/vllm/_moe_C.abi3.so",
    ".venv-vllm/lib/python3.12/site-packages/vllm/_flashmla_extension_C.abi3.so",
]

repo = Path(__file__).parent.parent
total = 0
for rel in SO_FILES:
    p = repo / rel
    if not p.exists():
        print(f"  MISSING: {p}")
        continue
    n = patch_file(p)
    total += n

print(f"\nDone. Total vna_hash fields patched: {total}")
