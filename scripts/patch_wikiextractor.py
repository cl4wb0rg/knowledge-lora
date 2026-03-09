#!/usr/bin/env python3
"""
Patch wikiextractor/extract.py for Python 3.12 compatibility.

Python 3.12 rejects (?i) flags embedded mid-pattern in re.compile().
This script moves them to the front of the affected patterns.
Idempotent: re-running after the patch is already applied is a no-op.
"""

import sys
from pathlib import Path

FIXES = [
    # ExtLinkBracketedRegex: '\[(((?i)...' → '(?i)\[((...'
    (
        r"'\[(((?i)' + '|'.join(wgUrlProtocols)",
        r"'(?i)\[((' + '|'.join(wgUrlProtocols)",
    ),
    # EXT_IMAGE_REGEX: raw string has ((?i)gif|...) at the end
    (
        r"/([A-Za-z0-9_.,~%\-+&;#*?!=()@\x80-\xFF]+)\.((?i)gif|png|jpg|jpeg)$",
        r"/([A-Za-z0-9_.,~%\-+&;#*?!=()@\x80-\xFF]+)\.(gif|png|jpg|jpeg)$",
    ),
    # EXT_IMAGE_REGEX: move (?i) to the front of the raw docstring pattern
    (
        'r"""^(http://|https://)',
        'r"""(?i)^(http://|https://)',
    ),
]


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path/to/extract.py>", file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1])
    src = path.read_text(encoding="utf-8")
    original = src

    for old, new in FIXES:
        src = src.replace(old, new)

    if src == original:
        print("Already patched — no changes made.")
        sys.exit(0)

    path.write_text(src, encoding="utf-8")
    print(f"Patched: {path}")


if __name__ == "__main__":
    main()
