#!/usr/bin/env python3
"""
Download the latest Wikipedia dump for one or more languages and verify checksums.

Usage:
    python scripts/01_download_wiki.py --lang de
    python scripts/01_download_wiki.py --lang de --lang en
"""
import hashlib
import logging
import re
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

DUMP_BASE_URL = "https://dumps.wikimedia.org"
LANG_RE = re.compile(r"^[a-z]{2,5}$")


def _validate_lang(lang: str) -> str:
    if not LANG_RE.match(lang):
        raise ValueError(
            f"Invalid language code {lang!r}. Expected 2-5 lowercase ASCII letters."
        )
    return lang


def _dump_url(lang: str) -> tuple[str, str]:
    wiki = f"{lang}wiki"
    filename = f"{wiki}-latest-pages-articles.xml.bz2"
    url = f"{DUMP_BASE_URL}/{wiki}/latest/{filename}"
    return url, filename


def _md5sums_url(lang: str) -> str:
    wiki = f"{lang}wiki"
    return f"{DUMP_BASE_URL}/{wiki}/latest/{wiki}-latest-md5sums.txt"


def _fetch_expected_md5(lang: str, filename: str) -> str | None:
    """Fetch the published MD5 checksum for *filename* from Wikimedia."""
    url = _md5sums_url(lang)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            for line in resp.read().decode().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == filename:
                    return parts[0]
    except Exception as exc:
        log.warning("Could not fetch checksums from %s: %s", url, exc)
    return None


def _md5_file(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324  (used for integrity, not security)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()



def download_dump(lang: str, out_dir: Path) -> None:
    """Download and verify the Wikipedia dump for *lang*."""
    _validate_lang(lang)
    url, filename = _dump_url(lang)
    dest = out_dir / filename
    tmp = dest.with_suffix(".bz2.tmp")

    if dest.exists():
        log.info("Already exists, skipping: %s", dest)
        return

    # Fetch expected checksum before starting the download
    log.info("Fetching expected checksum for %s …", filename)
    expected_md5 = _fetch_expected_md5(lang, filename)
    if expected_md5:
        log.info("Expected MD5: %s", expected_md5)
    else:
        log.warning("No checksum available — integrity will not be verified.")

    log.info("Downloading %s", url)
    log.info("  → %s", dest)

    # Resume partial download if .tmp already exists (e.g. after a stalled connection).
    resume_from = tmp.stat().st_size if tmp.exists() else 0
    if resume_from:
        log.info("  Resuming from %d MB already downloaded", resume_from // 1_048_576)

    try:
        req = urllib.request.Request(url)  # noqa: S310
        if resume_from:
            req.add_header("Range", f"bytes={resume_from}-")
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            total_size = int(resp.headers.get("Content-Length", 0)) + resume_from
            downloaded = resume_from
            mode = "ab" if resume_from else "wb"
            with tmp.open(mode) as f:
                while True:
                    chunk = resp.read(1 << 20)  # 1 MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = min(downloaded * 100 / total_size, 100)
                        print(
                            f"\r  {pct:5.1f}%  {downloaded/1_048_576:,.0f}"
                            f" / {total_size/1_048_576:,.0f} MB",
                            end="",
                            flush=True,
                        )
        print()  # newline after progress bar
    except Exception:
        # Leave the .tmp file in place so a retry can resume
        raise

    # Verify checksum
    if expected_md5:
        log.info("Verifying checksum …")
        actual = _md5_file(tmp)
        if actual != expected_md5:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {filename}:\n"
                f"  expected: {expected_md5}\n"
                f"  actual:   {actual}"
            )
        log.info("Checksum OK.")

    # Atomic rename
    tmp.rename(dest)
    log.info("Saved: %s", dest)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download Wikipedia XML dumps")
    parser.add_argument(
        "--lang",
        action="append",
        default=[],
        metavar="LANG",
        help="Language code (2-5 lowercase letters). Repeatable.",
    )
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()

    langs = args.lang or ["de"]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for lang in langs:
        try:
            download_dump(lang, out_dir)
        except Exception as exc:
            log.error("Failed to download %s: %s", lang, exc)
            sys.exit(1)

    log.info("All downloads complete.")


if __name__ == "__main__":
    main()
