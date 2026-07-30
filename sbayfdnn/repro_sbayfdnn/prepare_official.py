#!/usr/bin/env python3
"""Fetch and verify the pinned sBayFDNN code and arXiv source."""

from __future__ import annotations

import hashlib
import subprocess
import tarfile
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
OFFICIAL = HERE.parent / "official"
CODE = OFFICIAL / "code"
PAPER = OFFICIAL / "paper"
ARCHIVE = OFFICIAL / "paper-source.tar"
REPO = "https://github.com/mengyunwu2020/sBayFDNN.git"
COMMIT = "276d87949c8b973bf8b6748aa8df4f56086e063e"
ARXIV_URL = "https://arxiv.org/e-print/2602.20651"
ARCHIVE_SHA256 = "f9dae2ec3769e93f5887eca672e0d94c04fcf41b50580c0dc37a070ddeef2ccf"
TEX_SHA256 = "dc998374c660f66877c23afe3e703d2e121a85546d85e3bae5ff8faf5c252a08"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def main() -> int:
    OFFICIAL.mkdir(parents=True, exist_ok=True)
    if not CODE.exists():
        run("git", "clone", REPO, str(CODE))
        run("git", "checkout", "--detach", COMMIT, cwd=CODE)
    observed_commit = run("git", "rev-parse", "HEAD", cwd=CODE)
    if observed_commit != COMMIT:
        raise RuntimeError(f"Code revision mismatch: {observed_commit}")

    if not ARCHIVE.exists():
        urllib.request.urlretrieve(ARXIV_URL, ARCHIVE)
    if sha256(ARCHIVE) != ARCHIVE_SHA256:
        raise RuntimeError("arXiv source archive SHA-256 mismatch")
    if not PAPER.exists():
        PAPER.mkdir()
        with tarfile.open(ARCHIVE) as source:
            source.extractall(PAPER, filter="data")
    tex = PAPER / "ICML-FDNN.tex"
    if sha256(tex) != TEX_SHA256:
        raise RuntimeError("Paper TeX SHA-256 mismatch")
    print(f"code={observed_commit}")
    print(f"paper_tex_sha256={sha256(tex)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
