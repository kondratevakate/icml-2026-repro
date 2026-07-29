#!/usr/bin/env python3
"""Validate the frozen PyHealth evidence file and expected claim outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "evidence",
        nargs="?",
        type=Path,
        default=SCRIPT_DIR / "evidence" / "claims_audit.json",
    )
    args = parser.parse_args()
    data = json.loads(args.evidence.read_text(encoding="utf-8"))

    assert data["paper"]["openreview_id"] == "gMLVFN9hl8"
    assert data["provenance"]["official_repo_tag"] == "v2.0.1"
    assert (
        data["provenance"]["arxiv_source_archive_sha256"]
        == "8f592e99599e609ced3216700c1d4e878baffcb792c35a63a9c767c8224d078c"
    )
    assert (
        data["provenance"]["official_repo_commit"]
        == "ed562121b5bae185b36322c64ce6215c2095dd50"
    )
    assert data["provenance"]["official_repo_clean"] is True
    assert data["claims"]["C1"]["verdict"] == "VERIFIED"
    assert all(data["claims"]["C1"]["checks"].values())
    assert data["claims"]["C4"]["verdict"] == "FALSIFIED"
    assert (
        data["claims"]["C4"]["observed_mortality_loc"]["PyHealth 2.0"] == 34
    )
    assert (
        data["claims"]["C4"]["observed_mortality_loc"]["PyHealth 1.16"] == 27
    )
    assert data["claims"]["C4"]["observed_mortality_loc"]["Pandas"] == 51
    assert data["claims"]["C5"]["verdict"] == "INCONCLUSIVE"

    print(f"validated {args.evidence}")
    print(f"sha256 {sha256(args.evidence)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
