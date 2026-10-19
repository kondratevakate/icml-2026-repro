#!/usr/bin/env python3
"""Numerical primal-dual audit of MDCP Theorem 2.

The audit solves the randomized-set relaxation in Eq. (4) directly and checks
the superlevel-set and complementary-slackness characterization against the
LP dual returned by HiGHS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, minimize_scalar


def solve_randomized_set(
    densities: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, float]:
    n_sources, n_outcomes = densities.shape
    result = linprog(
        c=np.ones(n_outcomes),
        A_ub=-densities,
        b_ub=-(1.0 - alpha) * np.ones(n_sources),
        bounds=[(0.0, 1.0)] * n_outcomes,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    multipliers = -np.asarray(result.ineqlin.marginals)
    return np.asarray(result.x), multipliers, float(result.fun)


def dual_value(
    multipliers: np.ndarray, densities: np.ndarray, alpha: float
) -> float:
    shared_score = multipliers @ densities
    return float(
        (1.0 - alpha) * multipliers.sum()
        - np.maximum(shared_score - 1.0, 0.0).sum()
    )


def best_uniform_dual(densities: np.ndarray, alpha: float) -> float:
    def objective(scale: float) -> float:
        multipliers = np.full(densities.shape[0], scale)
        return -dual_value(multipliers, densities, alpha)

    result = minimize_scalar(objective, bounds=(0.0, 200.0), method="bounded")
    if not result.success:
        raise RuntimeError("uniform-direction dual optimization failed")
    return float(-result.fun)


def run_audit(
    seed: int,
    n_instances: int,
    n_sources: int,
    n_outcomes: int,
    alpha: float,
) -> dict:
    rng = np.random.default_rng(seed)
    tolerance = 2e-7
    maximum_gap = 0.0
    maximum_complementarity_error = 0.0
    maximum_structural_violation = 0.0
    minimum_coverage_margin = np.inf
    instances_with_active_source = 0
    uniform_direction_strictly_suboptimal = 0
    boundary_outcome_counts: list[int] = []

    for _ in range(n_instances):
        concentrations = np.exp(rng.normal(0.0, 1.0, size=(n_sources, n_outcomes)))
        densities = np.vstack([rng.dirichlet(row) for row in concentrations])
        randomized_set, multipliers, primal_value = solve_randomized_set(
            densities, alpha
        )
        shared_score = multipliers @ densities
        coverage = densities @ randomized_set
        dual = dual_value(multipliers, densities, alpha)

        maximum_gap = max(maximum_gap, abs(primal_value - dual))
        minimum_coverage_margin = min(
            minimum_coverage_margin, float(np.min(coverage - (1.0 - alpha)))
        )
        maximum_complementarity_error = max(
            maximum_complementarity_error,
            float(np.max(np.abs(multipliers * (coverage - (1.0 - alpha))))),
        )

        at_zero = randomized_set <= tolerance
        at_one = randomized_set >= 1.0 - tolerance
        fractional = ~(at_zero | at_one)
        violations = [
            np.max(np.maximum(shared_score[at_zero] - 1.0, 0.0), initial=0.0),
            np.max(np.maximum(1.0 - shared_score[at_one], 0.0), initial=0.0),
            np.max(np.abs(shared_score[fractional] - 1.0), initial=0.0),
        ]
        maximum_structural_violation = max(
            maximum_structural_violation, float(max(violations))
        )
        boundary_outcome_counts.append(int(np.sum(np.abs(shared_score - 1.0) < 1e-6)))

        if np.any(multipliers > tolerance):
            instances_with_active_source += 1
        if dual - best_uniform_dual(densities, alpha) > 1e-6:
            uniform_direction_strictly_suboptimal += 1

    passed = (
        maximum_gap < 1e-7
        and maximum_complementarity_error < 1e-7
        and maximum_structural_violation < 1e-6
        and minimum_coverage_margin > -1e-8
        and instances_with_active_source == n_instances
    )
    return {
        "status": "passed" if passed else "failed",
        "paper_anchor": "Theorem 2; Section 3.1; Eqs. (4)-(5)",
        "seed": seed,
        "n_instances": n_instances,
        "n_sources": n_sources,
        "n_outcomes": n_outcomes,
        "alpha": alpha,
        "maximum_primal_dual_gap": maximum_gap,
        "minimum_coverage_margin": minimum_coverage_margin,
        "maximum_complementarity_error": maximum_complementarity_error,
        "maximum_superlevel_structure_violation": maximum_structural_violation,
        "instances_with_at_least_one_active_source": instances_with_active_source,
        "uniform_direction_strictly_suboptimal_instances": (
            uniform_direction_strictly_suboptimal
        ),
        "boundary_outcomes": {
            "minimum": int(np.min(boundary_outcome_counts)),
            "median": float(np.median(boundary_outcome_counts)),
            "maximum": int(np.max(boundary_outcome_counts)),
        },
        "interpretation": (
            "The independently solved primal and paper dual agree; h_lambda > 1 "
            "selects included outcomes, h_lambda < 1 excludes them, and only "
            "the h_lambda = 1 boundary can be randomized. Active source "
            "constraints satisfy complementary slackness."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=260102998)
    parser.add_argument("--instances", type=int, default=500)
    parser.add_argument("--sources", type=int, default=3)
    parser.add_argument("--outcomes", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run_audit(
        args.seed, args.instances, args.sources, args.outcomes, args.alpha
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
