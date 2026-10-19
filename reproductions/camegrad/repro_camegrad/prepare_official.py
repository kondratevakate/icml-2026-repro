#!/usr/bin/env python3
"""Fetch the exact public artifacts audited by this reproduction."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
OFFICIAL = HERE.parent / "official"
CODE = OFFICIAL / "code"
SOURCE = OFFICIAL / "source"
ARCHIVE = OFFICIAL / "source.tar"
COMMIT = "79059e39060d13ef6b6cb2ea9ad7a5d8e2519c83"
ARCHIVE_SHA256 = (
    "9f5109704136930a9453e0995aad73ec6f794f51ff15c55e54a3f031369199d1"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    OFFICIAL.mkdir(parents=True, exist_ok=True)
    if not CODE.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/vpsg-research/CAME-Grad", str(CODE)],
            check=True,
        )
    subprocess.run(["git", "-C", str(CODE), "fetch", "origin", COMMIT], check=True)
    subprocess.run(["git", "-C", str(CODE), "checkout", "--detach", COMMIT], check=True)

    if not ARCHIVE.exists():
        urllib.request.urlretrieve(
            "https://export.arxiv.org/e-print/2605.22635v2", ARCHIVE
        )
    observed = digest(ARCHIVE)
    if observed != ARCHIVE_SHA256:
        raise RuntimeError(f"Unexpected arXiv archive SHA-256: {observed}")

    if SOURCE.exists():
        shutil.rmtree(SOURCE)
    SOURCE.mkdir()
    with tarfile.open(ARCHIVE) as handle:
        handle.extractall(SOURCE, filter="data")
    print(f"Prepared author commit {COMMIT} and verified arXiv source archive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
