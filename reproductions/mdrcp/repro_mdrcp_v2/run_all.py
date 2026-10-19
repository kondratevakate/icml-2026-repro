#!/usr/bin/env python3
"""Run the independent MDRCP theorem audits."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path) -> None:
    print(f"\n$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    root = Path(__file__).resolve().parent
    results = root / "results"
    results.mkdir(exist_ok=True)
    python = sys.executable
    run(
        [
            python,
            "audit_theorem2.py",
            "--output",
            str(results / "theorem2.json"),
        ],
        root,
    )
    run(
        [
            python,
            "audit_theorem3.py",
            "--output",
            str(results / "theorem3.json"),
        ],
        root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
