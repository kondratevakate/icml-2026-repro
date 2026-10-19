#!/usr/bin/env python3
"""Run the complete local ROCP audit bundle."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path) -> None:
    print(f"\n$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-repo",
        type=Path,
        help="Optional checkout of the authors' official repository",
    )
    parser.add_argument("--random-cases", type=int, default=120)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    results = root / "results"
    results.mkdir(exist_ok=True)
    python = sys.executable

    run([python, "-m", "unittest", "discover", "-s", "tests", "-v"], root)
    run(
        [
            python,
            "audit_claim1.py",
            "--random-cases",
            str(args.random_cases),
            "--output",
            str(results / "claim1.json"),
        ],
        root,
    )
    run(
        [
            python,
            "audit_claim3_exchangeability.py",
            "--max-sample-size",
            "6",
            "--output",
            str(results / "claim3_exchangeability.json"),
        ],
        root,
    )
    if args.official_repo:
        run(
            [
                python,
                "audit_official_divergence.py",
                "--official-repo",
                str(args.official_repo),
                "--output",
                str(results / "official_divergence.json"),
            ],
            root,
        )
    else:
        print("\nSkipped official-code comparison: pass --official-repo to enable it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
