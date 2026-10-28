from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMMANDS = [
    ["audit_three_scales.py", "--output", "results/three_scales.json"],
    ["audit_scalability_proxy.py", "--output", "results/scalability_proxy.json"],
]


def main() -> None:
    for command in COMMANDS:
        full_command = [sys.executable, *command]
        print(f"\n$ {' '.join(full_command)}", flush=True)
        subprocess.run(full_command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
