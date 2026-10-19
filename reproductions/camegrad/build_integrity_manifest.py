#!/usr/bin/env python3
"""Create a deterministic SHA-256 manifest for the publishable candidate."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "INTEGRITY_MANIFEST.sha256"
EXCLUDED_PARTS = {"official", "__pycache__"}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def manifest_lines() -> list[str]:
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != OUTPUT
        and not any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts)
        and path.suffix != ".pyc"
    )
    return [
        f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in files
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    lines = manifest_lines()
    rendered = "\n".join(lines) + "\n"
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="ascii") != rendered:
            print("Integrity manifest mismatch")
            return 1
        print(f"Verified {len(lines)} integrity entries")
        return 0

    OUTPUT.write_text(rendered, encoding="ascii")
    print(f"Wrote {OUTPUT} with {len(lines)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
