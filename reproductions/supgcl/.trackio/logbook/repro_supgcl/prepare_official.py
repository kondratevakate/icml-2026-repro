#!/usr/bin/env python3
"""Fetch and verify the pinned SupGCL code and arXiv source."""

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
REPO = "https://github.com/shobioinfo/SupGCL.git"
COMMIT = "a384eeaeffaabfd140335e19a0633ab400ba3bcb"
ARXIV_URL = "https://arxiv.org/e-print/2505.17786"
ARCHIVE_SHA256 = "20d8c3ef3ce61e816e88817d2cccabddd7b0fc8e8d3b0c3fa2ba1436b5e55ec8"
METHODOLOGY_SHA256 = "003031f7f1941123c63eb60c4ad81a1f06376eb41c311dfa6f01c05cbc884fd1"
PROOF_SHA256 = "b1faceae982630f9f47f2e8dbfe96f196a4dbd632c301cbdbb96f83c30e467d6"


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

    methodology = PAPER / "04Methodology.tex"
    proof = PAPER / "A01ProofOfTheorem.tex"
    if sha256(methodology) != METHODOLOGY_SHA256:
        raise RuntimeError("Methodology TeX SHA-256 mismatch")
    if sha256(proof) != PROOF_SHA256:
        raise RuntimeError("Theorem proof TeX SHA-256 mismatch")
    print(f"code={observed_commit}")
    print(f"methodology_sha256={sha256(methodology)}")
    print(f"proof_sha256={sha256(proof)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
