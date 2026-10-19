#!/usr/bin/env python3
"""Fetch the pinned public artifacts needed by the reproduction."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download


REPOSITORIES = {
    "GlucoSim": (
        "https://github.com/safe-autonomy-lab/GlucoSim.git",
        "f5662ccca607f11fd3bf30fde513ae0cf2ce420e",
    ),
    "GlucoAlg": (
        "https://github.com/safe-autonomy-lab/GlucoAlg.git",
        "50d3134fd6f0e01019295be0d6b99f110f794753",
    ),
}
MODEL_REPO = "safe-diabetes-benchmark/safe-diabetes-t1d-adolescent-cpo"
MODEL_REVISION = "993c43697d18333247e5d1fbe39ba37b989393ee"
CHECKPOINT_SHA256 = (
    "806fe99618dc15549eb3a543b1464329f538f30cbb6ca0eecc8ab07058dac1c8"
)


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path("../official"))
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    for name, (url, revision) in REPOSITORIES.items():
        checkout = destination / name
        if not checkout.exists():
            run("git", "clone", "--filter=blob:none", url, str(checkout))
        run("git", "fetch", "origin", revision, cwd=checkout)
        run("git", "checkout", "--detach", revision, cwd=checkout)

    model_root = destination / "models" / "t1d-adolescent-cpo"
    for filename in (
        "checkpoints/seed0/epoch-2441.pt",
        "config/seed0/config.json",
    ):
        downloaded = Path(
            hf_hub_download(
            repo_id=MODEL_REPO,
            filename=filename,
            revision=MODEL_REVISION,
            local_dir=model_root,
            )
        )
        if filename.endswith(".pt"):
            digest = hashlib.sha256(downloaded.read_bytes()).hexdigest()
            if digest != CHECKPOINT_SHA256:
                raise RuntimeError(
                    f"Checkpoint SHA-256 mismatch: {digest} != {CHECKPOINT_SHA256}"
                )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
