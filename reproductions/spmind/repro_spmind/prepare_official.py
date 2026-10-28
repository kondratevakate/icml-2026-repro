#!/usr/bin/env python3
"""Fetch and pin the public SP-Mind artifacts needed by the audit."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path


REPO_URL = "https://github.com/tomtommyyuan/spmind.git"
COMMIT = "d5b889c3649fbdfd1489e5b2f60fc52a0d3ddbc6"
ARXIV_URL = "https://arxiv.org/e-print/2606.24235"
SOURCE_SHA256 = "2b38444fea77fe5b03f74fc682771785294f63ce20909dfd58adeaee9e8cd041"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def run(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def safe_extract(archive: Path, destination: Path) -> None:
    destination_abs = destination.resolve()
    with tarfile.open(archive, "r:*") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if destination_abs not in target.parents and target != destination_abs:
                raise ValueError(f"Unsafe archive member: {member.name}")
        handle.extractall(destination)


def prepare(root: Path) -> None:
    code = root / "code"
    source = root / "arxiv-source"
    root.mkdir(parents=True, exist_ok=True)

    if not code.exists():
        run("git", "clone", "--no-checkout", REPO_URL, str(code))
        run("git", "checkout", "--detach", COMMIT, cwd=code)
    actual_commit = run("git", "rev-parse", "HEAD", cwd=code)
    if actual_commit != COMMIT:
        raise RuntimeError(f"Expected code commit {COMMIT}, found {actual_commit}")

    required_source = source / "resources" / "sections" / "experiments.tex"
    if not required_source.exists():
        source.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "source.tar"
            urllib.request.urlretrieve(ARXIV_URL, archive)
            actual_hash = digest(archive)
            if actual_hash != SOURCE_SHA256:
                raise RuntimeError(
                    f"Expected arXiv source SHA-256 {SOURCE_SHA256}, found {actual_hash}"
                )
            safe_extract(archive, source)

    print(f"code={actual_commit}")
    print(f"source={source}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "official",
    )
    args = parser.parse_args()
    prepare(args.official_root)


if __name__ == "__main__":
    main()
