#!/usr/bin/env python3
"""Run the data-independent MLCI audits."""

from __future__ import annotations

import subprocess
import sys


def run(script: str) -> None:
    print(f"\n$ {sys.executable} {script}", flush=True)
    subprocess.run([sys.executable, script], check=True)


def main() -> None:
    run("audit_nhsic_method.py")
    run("audit_rank_one_threshold.py")
    run("test_build_mimic_cohort.py")
    print("\nMLCI C1/C5 audits and cohort integration tests passed.", flush=True)


if __name__ == "__main__":
    main()
