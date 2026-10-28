#!/usr/bin/env python3
"""Exhaustive finite exchangeability audit for paper Algorithm 1."""

from __future__ import annotations

import argparse
import json
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

from rocp_core import DiscreteROCP


LOSS_MATRIX = np.asarray(
    [
        [0.0, 3.0, 8.0],
        [2.0, 0.0, 5.0],
        [20.0, 4.0, 0.0],
    ]
)

FEATURE_PROBABILITIES = (
    (0.75, 0.20, 0.05),
    (0.15, 0.70, 0.15),
    (0.05, 0.25, 0.70),
)

# Nine possible (x, y) observations. Every multiset of these observations is
# an exchangeability orbit; a general finite exchangeable law is a mixture of
# uniform distributions over such orbits.
OBSERVATION_SUPPORT = tuple(
    (feature, label)
    for feature in range(len(FEATURE_PROBABILITIES))
    for label in range(LOSS_MATRIX.shape[0])
)


def orbit_coverage(
    orbit: tuple[int, ...],
    alpha: float,
    candidate_betas: tuple[float, ...],
    covered: np.ndarray,
) -> tuple[float, float]:
    # For the true candidate label, every held-out problem augments back to the
    # same full orbit. Therefore beta is common across test positions.
    target = 1.0 - alpha
    for beta_index, beta in enumerate(candidate_betas):
        hits = sum(covered[beta_index, observation] for observation in orbit)
        coverage = float(hits / len(orbit))
        if coverage + 1e-11 >= target:
            return coverage, beta
    raise AssertionError("the full-set endpoint should always be feasible")


def run_audit(max_sample_size: int) -> dict[str, object]:
    predictor = DiscreteROCP(LOSS_MATRIX)
    candidate_betas = {
        beta
        for probabilities in FEATURE_PROBABILITIES
        for beta in predictor.beta_breakpoints(probabilities)
    }
    # Include an endpoint strictly beyond every affine intersection.
    candidate_betas.add(max(candidate_betas, default=0.0) + 1.0)
    beta_grid = tuple(sorted(candidate_betas))
    covered = np.zeros((len(beta_grid), len(OBSERVATION_SUPPORT)), dtype=np.int8)
    for beta_index, beta in enumerate(beta_grid):
        for observation_index, (feature, label) in enumerate(OBSERVATION_SUPPORT):
            covered[beta_index, observation_index] = int(
                label in predictor.predict_set(FEATURE_PROBABILITIES[feature], beta)
            )

    alpha_grid = {
        3: (0.10, 1 / 3, 0.50),
        4: (0.10, 0.25, 0.50),
        5: (0.10, 0.20, 0.40),
        6: (0.10, 1 / 6, 1 / 3),
    }
    summaries = []
    total_orbit_alpha_checks = 0

    for sample_size in range(3, max_sample_size + 1):
        worst_margin = np.inf
        worst_coverage = 1.0
        worst_case = None
        orbit_count = 0
        for orbit in combinations_with_replacement(
            range(len(OBSERVATION_SUPPORT)), sample_size
        ):
            orbit_count += 1
            for alpha in alpha_grid[sample_size]:
                coverage, beta = orbit_coverage(
                    orbit, alpha, beta_grid, covered
                )
                margin = coverage - (1.0 - alpha)
                if margin < worst_margin:
                    worst_margin = margin
                    worst_coverage = coverage
                    worst_case = {
                        "orbit": list(orbit),
                        "alpha": alpha,
                        "beta": beta,
                    }
                if margin < -1e-10:
                    raise AssertionError(
                        f"coverage failure: N={sample_size}, orbit={orbit}, "
                        f"alpha={alpha}, coverage={coverage}, beta={beta}"
                    )
                total_orbit_alpha_checks += 1
        summaries.append(
            {
                "sample_size_n_plus_1": sample_size,
                "exchangeability_orbits": orbit_count,
                "alphas": list(alpha_grid[sample_size]),
                "worst_coverage": worst_coverage,
                "worst_margin_above_bound": float(worst_margin),
                "worst_case": worst_case,
            }
        )

    # Directly exercise the literal candidate-label loop, not just its
    # true-label simplification used in the exhaustive proof audit.
    calibration_probabilities = [
        FEATURE_PROBABILITIES[0],
        FEATURE_PROBABILITIES[1],
        FEATURE_PROBABILITIES[2],
    ]
    calibration_labels = [0, 1, 2]
    direct_set = predictor.algorithm1_prediction_set(
        calibration_probabilities,
        calibration_labels,
        FEATURE_PROBABILITIES[0],
        alpha=0.25,
    )
    if not direct_set:
        raise AssertionError("Algorithm 1 returned an empty direct prediction set")

    return {
        "status": "passed",
        "loss_matrix_labels_by_actions": LOSS_MATRIX.tolist(),
        "feature_probabilities": [list(row) for row in FEATURE_PROBABILITIES],
        "observation_support_size": len(OBSERVATION_SUPPORT),
        "total_orbit_alpha_checks": total_orbit_alpha_checks,
        "summaries": summaries,
        "direct_algorithm1_example": {
            "alpha": 0.25,
            "prediction_set": sorted(direct_set),
        },
        "interpretation": (
            "Every finite exchangeable law on this support is a mixture over "
            "the audited permutation orbits, so the bound follows orbit-wise."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sample-size", type=int, choices=(3, 4, 5, 6), default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_audit(args.max_sample_size)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
