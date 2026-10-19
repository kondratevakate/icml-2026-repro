#!/usr/bin/env python3
"""Finite-alphabet convergence audit of MDCP Theorem 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.stats import spearmanr


def solve_dual_weights(densities: np.ndarray, alpha: float) -> np.ndarray:
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
    return -np.asarray(result.ineqlin.marginals)


def estimate_probabilities(
    rng: np.random.Generator,
    true_probabilities: np.ndarray,
    n: int,
    smoothing: float,
) -> np.ndarray:
    rows = []
    for probabilities in true_probabilities:
        counts = rng.multinomial(n, probabilities)
        rows.append((counts + smoothing) / (n + smoothing * len(counts)))
    return np.vstack(rows)


def conservative_max_p_set(
    shared_score: np.ndarray,
    calibration_labels: list[np.ndarray],
    alpha: float,
) -> np.ndarray:
    candidate_nonconformity = -shared_score
    p_max = np.zeros_like(shared_score)
    for labels in calibration_labels:
        calibration_scores = candidate_nonconformity[labels]
        p_values = np.array(
            [
                (1.0 + np.sum(calibration_scores >= score))
                / (1.0 + len(calibration_scores))
                for score in candidate_nonconformity
            ]
        )
        p_max = np.maximum(p_max, p_values)
    return p_max >= alpha


def one_estimate(
    true_probabilities: np.ndarray,
    oracle_weights: np.ndarray,
    oracle_score: np.ndarray,
    oracle_set: np.ndarray,
    n: int,
    seed: int,
    alpha: float,
    smoothing: float,
    frozen_weights: np.ndarray | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    n_estimation = n // 2
    n_calibration = n - n_estimation
    estimated = estimate_probabilities(
        rng, true_probabilities, n_estimation, smoothing
    )
    weights = solve_dual_weights(estimated, alpha)
    shared_score = weights @ estimated
    plug_in_set = shared_score >= 1.0
    calibration = [
        rng.choice(
            true_probabilities.shape[1],
            size=n_calibration,
            p=true_probabilities[k],
        )
        for k in range(true_probabilities.shape[0])
    ]
    conformal_set = conservative_max_p_set(shared_score, calibration, alpha)

    row = {
        "sup_density_error": float(np.max(np.abs(estimated - true_probabilities))),
        "sup_dual_weight_error": float(np.max(np.abs(weights - oracle_weights))),
        "sup_shared_score_error": float(np.max(np.abs(shared_score - oracle_score))),
        "plugin_symmetric_difference": int(np.sum(plug_in_set != oracle_set)),
        "conformal_symmetric_difference": int(np.sum(conformal_set != oracle_set)),
        "conformal_set_size_gap": int(abs(np.sum(conformal_set) - np.sum(oracle_set))),
    }
    if frozen_weights is not None:
        frozen_score = frozen_weights @ estimated
        row["frozen_weight_sup_shared_score_error"] = float(
            np.max(np.abs(frozen_score - oracle_score))
        )
    return row


def run_audit(
    seed: int,
    seeds_per_size: int,
    n_sources: int,
    n_outcomes: int,
    alpha: float,
) -> dict:
    instance_rng = np.random.default_rng(seed)
    concentrations = np.exp(
        instance_rng.normal(0.0, 0.8, size=(n_sources, n_outcomes))
    )
    true_probabilities = np.vstack(
        [instance_rng.dirichlet(row) for row in concentrations]
    )
    oracle_weights = solve_dual_weights(true_probabilities, alpha)
    oracle_score = oracle_weights @ true_probabilities
    oracle_set = oracle_score >= 1.0
    boundary_size = int(np.sum(np.abs(oracle_score - 1.0) < 1e-7))

    frozen_rng = np.random.default_rng(seed + 1)
    frozen_estimate = estimate_probabilities(
        frozen_rng, true_probabilities, 50, smoothing=0.5
    )
    frozen_weights = solve_dual_weights(frozen_estimate, alpha)

    sample_sizes = [100, 250, 500, 1000, 2500, 5000, 10000, 25000]
    summaries = []
    for n in sample_sizes:
        rows = [
            one_estimate(
                true_probabilities,
                oracle_weights,
                oracle_score,
                oracle_set,
                n,
                seed + 10_000 * (i + 1) + n,
                alpha,
                smoothing=0.5,
                frozen_weights=frozen_weights,
            )
            for i in range(seeds_per_size)
        ]
        summaries.append(
            {
                "n_per_source": n,
                **{
                    key: float(np.mean([row[key] for row in rows]))
                    for key in rows[0]
                },
            }
        )

    log_n = np.log(np.asarray(sample_sizes, dtype=float))
    score_errors = np.asarray(
        [row["sup_shared_score_error"] for row in summaries]
    )
    density_errors = np.asarray([row["sup_density_error"] for row in summaries])
    weight_errors = np.asarray(
        [row["sup_dual_weight_error"] for row in summaries]
    )
    conformal_differences = np.asarray(
        [row["conformal_symmetric_difference"] for row in summaries]
    )
    frozen_errors = np.asarray(
        [row["frozen_weight_sup_shared_score_error"] for row in summaries]
    )
    score_rho = float(spearmanr(log_n, score_errors).statistic)
    density_rho = float(spearmanr(log_n, density_errors).statistic)
    weight_rho = float(spearmanr(log_n, weight_errors).statistic)
    set_rho = float(spearmanr(log_n, conformal_differences).statistic)

    passed = (
        density_errors[-1] < density_errors[0]
        and weight_errors[-1] < weight_errors[0]
        and score_errors[-1] < score_errors[0]
        and conformal_differences[-1] <= conformal_differences[0]
        and conformal_differences[-1] <= boundary_size
        and max(density_rho, weight_rho, score_rho) < -0.7
        and frozen_errors[-1] > score_errors[-1]
    )
    return {
        "status": "passed" if passed else "failed",
        "paper_anchor": "Theorem 3; Section 3.2; Eqs. (6)-(7)",
        "seed": seed,
        "seeds_per_sample_size": seeds_per_size,
        "n_sources": n_sources,
        "n_outcomes": n_outcomes,
        "alpha": alpha,
        "oracle_boundary_measure_count": boundary_size,
        "oracle_set_size": int(np.sum(oracle_set)),
        "summaries": summaries,
        "spearman_log_n_vs_density_error": density_rho,
        "spearman_log_n_vs_dual_weight_error": weight_rho,
        "spearman_log_n_vs_shared_score_error": score_rho,
        "spearman_log_n_vs_conformal_symmetric_difference": set_rho,
        "largest_n_error_ratio_vs_smallest": float(score_errors[-1] / score_errors[0]),
        "frozen_control_error_ratio_at_largest_n": float(
            frozen_errors[-1] / score_errors[-1]
        ),
        "interpretation": (
            "Estimated densities and dual weights converge to the oracle shared "
            "score, while the calibrated max-p set approaches the oracle set up "
            "to its boundary. Freezing the dual weights breaks the consistency "
            "precondition and leaves a "
            f"{frozen_errors[-1] / score_errors[-1]:.2f}x larger score error at "
            "the largest audited sample size."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=260102999)
    parser.add_argument("--seeds-per-size", type=int, default=20)
    parser.add_argument("--sources", type=int, default=3)
    parser.add_argument("--outcomes", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run_audit(
        args.seed,
        args.seeds_per_size,
        args.sources,
        args.outcomes,
        args.alpha,
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
