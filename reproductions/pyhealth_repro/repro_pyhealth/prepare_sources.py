#!/usr/bin/env python3
"""Fetch and verify the exact public sources used by the PyHealth audit."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tarfile
import urllib.request
from pathlib import Path


ARXIV_URL = "https://export.arxiv.org/e-print/2601.16414v2"
ARXIV_SHA256 = (
    "8f592e99599e609ced3216700c1d4e878baffcb792c35a63a9c767c8224d078c"
)
REPO_URL = "https://github.com/sunlabuiuc/PyHealth.git"
REPO_TAG = "v2.0.1"
REPO_COMMIT = "ed562121b5bae185b36322c64ce6215c2095dd50"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"unsafe archive member: {member.name}")
        handle.extractall(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    args = parser.parse_args()
    root = args.root.resolve()
    paper = root / "paper_source"
    repo = root / "official_pyhealth"
    paper.mkdir(parents=True, exist_ok=True)

    archive = paper / "source.tar"
    if not archive.exists():
        print(f"downloading {ARXIV_URL}")
        urllib.request.urlretrieve(ARXIV_URL, archive)
    observed_archive_hash = sha256(archive)
    if observed_archive_hash != ARXIV_SHA256:
        raise ValueError(
            f"arXiv source hash mismatch: {observed_archive_hash}"
        )
    if not (paper / "example_paper.tex").exists():
        safe_extract(archive, paper)

    if not repo.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                REPO_TAG,
                REPO_URL,
                str(repo),
            ],
            check=True,
        )
    observed_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        encoding="utf-8",
    ).strip()
    if observed_commit != REPO_COMMIT:
        raise ValueError(f"official repository commit mismatch: {observed_commit}")
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"],
        text=True,
        encoding="utf-8",
    ).strip()
    if status:
        raise ValueError("official repository checkout is not clean")

    print(f"paper source sha256: {observed_archive_hash}")
    print(f"official repository commit: {observed_commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
