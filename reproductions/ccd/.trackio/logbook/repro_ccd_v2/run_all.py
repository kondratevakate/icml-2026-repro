from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMMANDS = [
    ["audit_metric_identities.py", "--output", "results/metric_identities.json"],
    ["audit_algorithm1_crossfit.py", "--output", "results/algorithm1.json"],
    [
        "audit_released_results.py",
        "--input-dir",
        "official_results",
        "--output",
        "results/released_results.json",
    ],
]


def main() -> None:
    for command in COMMANDS:
        full_command = [sys.executable, *command]
        print(f"\n$ {' '.join(full_command)}", flush=True)
        subprocess.run(full_command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
