#!/usr/bin/env python3
"""Executable comparison between paper Algorithm 1 and released rocp.py."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np

from audit_claim3_exchangeability import (
    FEATURE_PROBABILITIES,
    LOSS_MATRIX,
    OBSERVATION_SUPPORT,
)
from rocp_core import DiscreteROCP


def load_official_class(official_repo: Path):
    module_path = official_repo / "rocp.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"missing official implementation: {module_path}")
    spec = importlib.util.spec_from_file_location("official_rocp", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RiskOptimalConformalPredictor


def git_revision(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def run_audit(official_repo: Path, max_examples: int) -> dict[str, object]:
    OfficialPredictor = load_official_class(official_repo)
    official = OfficialPredictor(actions=(0, 1, 2), loss_matrix=LOSS_MATRIX)
    paper = DiscreteROCP(LOSS_MATRIX)

    probe_probabilities = FEATURE_PROBABILITIES[0]
    official_t_zero = frozenset(official.compute_set(probe_probabilities, 0.0))
    paper_t_zero = paper.pointwise_choice(probe_probabilities, 0.0).label_set
    if official_t_zero == paper_t_zero:
        raise AssertionError("expected the documented t=0 divergence")

    divergences = []
    for calibration_indices in itertools.product(
        range(len(OBSERVATION_SUPPORT)), repeat=2
    ):
        calibration_probabilities = [
            FEATURE_PROBABILITIES[OBSERVATION_SUPPORT[index][0]]
            for index in calibration_indices
        ]
        calibration_labels = [
            OBSERVATION_SUPPORT[index][1] for index in calibration_indices
        ]
        for test_feature in range(len(FEATURE_PROBABILITIES)):
            test_probabilities = FEATURE_PROBABILITIES[test_feature]
            for alpha in (1 / 3, 0.5):
                official_beta = official.calibrate_beta(
                    calibration_probabilities,
                    calibration_labels,
                    alpha,
                )
                official_set = frozenset(
                    official.predict_set(test_probabilities, official_beta)
                )
                paper_set = paper.algorithm1_prediction_set(
                    calibration_probabilities,
                    calibration_labels,
                    test_probabilities,
                    alpha,
                )
                if official_set != paper_set:
                    divergences.append(
                        {
                            "calibration_observation_indices": list(
                                calibration_indices
                            ),
                            "calibration_labels": calibration_labels,
                            "test_feature": test_feature,
                            "alpha": alpha,
                            "official_global_beta": official_beta,
                            "official_set": sorted(official_set),
                            "paper_algorithm1_set": sorted(paper_set),
                        }
                    )
                    if len(divergences) >= max_examples:
                        break
            if len(divergences) >= max_examples:
                break
        if len(divergences) >= max_examples:
            break

    if not divergences:
        raise AssertionError("search found no candidate-calibration divergence")

    return {
        "status": "passed",
        "official_repo": str(official_repo.resolve()),
        "official_git_revision": git_revision(official_repo),
        "t_zero_divergence": {
            "probabilities": list(probe_probabilities),
            "official_set": sorted(official_t_zero),
            "paper_remark_3_2_set": sorted(paper_t_zero),
        },
        "candidate_calibration_divergence_examples": divergences,
        "interpretation": (
            "These are implementation divergences, not evidence against the "
            "paper's finite-sample theorem."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True, type=Path)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_audit(args.official_repo, args.max_examples)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
