#!/usr/bin/env python3
"""Fetch and verify the exact arXiv artifacts used by this audit."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
OFFICIAL = HERE.parent / "official"
SOURCE = OFFICIAL / "source"
ARCHIVE = OFFICIAL / "source.tar"
PDF = OFFICIAL / "paper.pdf"
ARCHIVE_SHA256 = (
    "7862c54d7da9cabecfefbc7bd2308c7cdff507223be68ed9c9ebaada73fe1129"
)
PDF_SHA256 = (
    "adc0defaed5b133c8128b1ee3cb0962e07e6ab828700817855a3a6876f4fccaa"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    OFFICIAL.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        urllib.request.urlretrieve(
            "https://export.arxiv.org/e-print/2606.03332", ARCHIVE
        )
    if not PDF.exists():
        urllib.request.urlretrieve("https://arxiv.org/pdf/2606.03332", PDF)
    if digest(ARCHIVE) != ARCHIVE_SHA256:
        raise RuntimeError("Unexpected arXiv source archive SHA-256")
    if digest(PDF) != PDF_SHA256:
        raise RuntimeError("Unexpected arXiv PDF SHA-256")
    if SOURCE.exists():
        shutil.rmtree(SOURCE)
    SOURCE.mkdir()
    with tarfile.open(ARCHIVE) as handle:
        handle.extractall(SOURCE, filter="data")
    print("Verified arXiv source and PDF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

