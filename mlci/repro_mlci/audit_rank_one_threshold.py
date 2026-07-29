#!/usr/bin/env python3
"""Audit Lemmas 6.1-6.2 and Theorem 6.3 numerically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def centering(size: int) -> np.ndarray:
    return np.eye(size) - np.ones((size, size)) / size


def score_kernel(scores: np.ndarray, bandwidth: float = 0.8) -> np.ndarray:
    differences = scores[:, None] - scores[None, :]
    return np.exp(-(differences**2) / (2 * bandwidth**2))


def centered_kernel(scores: np.ndarray) -> np.ndarray:
    h = centering(len(scores))
    return h @ score_kernel(scores) @ h


def label_kernel(labels: np.ndarray) -> np.ndarray:
    return (labels[:, None] == labels[None, :]).astype(float)


def build_profiles(
    labels: np.ndarray, alpha: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray]]:
    h = centering(labels.shape[0])
    profiles = [h @ labels[:, task] for task in range(labels.shape[1])]
    rows = [
        np.sqrt(alpha[task]) * profile / np.linalg.norm(profile)
        for task, profile in enumerate(profiles)
    ]
    return np.stack(rows), profiles


def weighted_objective(
    kc: np.ndarray, profiles: list[np.ndarray], alpha: np.ndarray
) -> float:
    norm = np.linalg.norm(kc)
    if norm == 0:
        return 0.0
    return float(
        sum(
            alpha[task]
            * (profile @ kc @ profile)
            / (norm * (profile @ profile))
            for task, profile in enumerate(profiles)
        )
    )


def alignment_objective(kc: np.ndarray, matrix: np.ndarray) -> float:
    norm = np.linalg.norm(kc)
    if norm == 0:
        return 0.0
    return float(np.sum(kc * (matrix.T @ matrix)) / norm)


def threshold_score(size: int, split: int, separation: float) -> np.ndarray:
    scores = np.zeros(size)
    scores[split:] = separation
    return scores


def step_vector(size: int, split: int) -> np.ndarray:
    indicator = np.zeros(size)
    indicator[split:] = 1
    return centering(size) @ indicator


def floored_objective(
    kc: np.ndarray, matrix: np.ndarray, epsilon: float
) -> float:
    return float(
        np.sum(kc * (matrix.T @ matrix))
        / max(np.linalg.norm(kc), epsilon)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/rank_one_threshold.json")
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()

    maxima = {
        "delta_kernel_identity_error": 0.0,
        "weighted_alignment_error": 0.0,
        "rank_one_residual": 0.0,
        "shared_direction_centering_error": 0.0,
        "projected_reduction_error": 0.0,
        "threshold_formula_error": 0.0,
        "separation_invariance_error": 0.0,
        "global_bound_excess": 0.0,
        "near_optimality_excess": 0.0,
        "uniform_bound_excess": 0.0,
    }
    split_failures = 0

    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        size, tasks = 24 + seed % 9, 4
        base = rng.integers(0, 2, size=size).astype(float)
        if base.min() == base.max():
            base[:2] = [0, 1]
        labels = np.stack(
            [base, 1 - base, base, 1 - base], axis=1
        )
        alpha = rng.uniform(0.4, 1.8, size=tasks)
        matrix, profiles = build_profiles(labels, alpha)
        h = centering(size)

        for task, profile in enumerate(profiles):
            lc = h @ label_kernel(labels[:, task]) @ h
            expected = 2 * np.outer(profile, profile)
            maxima["delta_kernel_identity_error"] = max(
                maxima["delta_kernel_identity_error"],
                float(np.max(np.abs(lc - expected))),
            )

        _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
        direction = vt[0]
        sigma = singular_values[0]
        rank_one = sigma * np.outer(
            np.linalg.svd(matrix, full_matrices=False)[0][:, 0],
            direction,
        )
        maxima["rank_one_residual"] = max(
            maxima["rank_one_residual"],
            float(np.linalg.norm(matrix - rank_one)),
        )
        maxima["shared_direction_centering_error"] = max(
            maxima["shared_direction_centering_error"],
            abs(float(direction.sum())),
        )

        monotone = np.sort(rng.normal(size=size))
        kc = centered_kernel(monotone)
        direct = weighted_objective(kc, profiles, alpha)
        aligned = alignment_objective(kc, matrix)
        projected = alignment_objective(kc, rank_one)
        bar_delta = float(direction @ kc @ direction / np.linalg.norm(kc))
        maxima["weighted_alignment_error"] = max(
            maxima["weighted_alignment_error"], abs(direct - aligned)
        )
        maxima["projected_reduction_error"] = max(
            maxima["projected_reduction_error"],
            abs(projected - sigma**2 * bar_delta),
        )
        maxima["global_bound_excess"] = max(
            maxima["global_bound_excess"], bar_delta - 1.0
        )

        rho_squared = []
        full_values = []
        first_separation_values = []
        second_separation_values = []
        for split in range(1, size):
            step = step_vector(size, split)
            rho2 = float(
                (direction @ step) ** 2
                / ((direction @ direction) * (step @ step))
            )
            rho_squared.append(rho2)
            values = []
            for separation in (0.3, 1.2):
                threshold = threshold_score(size, split, separation)
                threshold_kc = centered_kernel(threshold)
                value = float(
                    direction
                    @ threshold_kc
                    @ direction
                    / np.linalg.norm(threshold_kc)
                )
                values.append(value)
                maxima["threshold_formula_error"] = max(
                    maxima["threshold_formula_error"], abs(value - rho2)
                )
            first_separation_values.append(values[0])
            second_separation_values.append(values[1])
            full_values.append(
                weighted_objective(
                    centered_kernel(threshold_score(size, split, 0.7)),
                    profiles,
                    alpha,
                )
            )

        maxima["separation_invariance_error"] = max(
            maxima["separation_invariance_error"],
            float(
                np.max(
                    np.abs(
                        np.asarray(first_separation_values)
                        - np.asarray(second_separation_values)
                    )
                )
            ),
        )
        rho_best = int(np.argmax(rho_squared))
        full_best = int(np.argmax(full_values))
        if (
            max(full_values) - full_values[rho_best] > 1e-10
            or max(rho_squared) - rho_squared[full_best] > 1e-10
        ):
            split_failures += 1

        random_labels = rng.integers(0, 2, size=(size, tasks)).astype(float)
        for task in range(tasks):
            if random_labels[:, task].min() == random_labels[:, task].max():
                random_labels[:2, task] = [0, 1]
        general, _ = build_profiles(random_labels, alpha)
        u, s, general_vt = np.linalg.svd(general, full_matrices=False)
        general_rank_one = s[0] * np.outer(u[:, 0], general_vt[0])
        candidates = [
            threshold_score(size, split, 0.7)
            for split in range(1, size)
        ]
        candidates.extend(
            np.sort(rng.normal(size=size)) for _ in range(20)
        )
        epsilon = 1e-3
        full_scores = []
        projected_scores = []
        gaps = []
        for candidate in candidates:
            candidate_kc = centered_kernel(candidate)
            full_score = floored_objective(
                candidate_kc, general, epsilon
            )
            projected_score = floored_objective(
                candidate_kc, general_rank_one, epsilon
            )
            full_scores.append(full_score)
            projected_scores.append(projected_score)
            gaps.append(abs(full_score - projected_score))
        gap = max(gaps)
        selected = int(np.argmax(projected_scores))
        near_optimality_excess = (
            max(full_scores) - 2 * gap - full_scores[selected]
        )
        maxima["near_optimality_excess"] = max(
            maxima["near_optimality_excess"], near_optimality_excess
        )

        residual = general - general_rank_one
        explicit_bound = sum(
            (
                np.linalg.norm(general[task], 1)
                + np.linalg.norm(general_rank_one[task], 1)
            )
            * np.linalg.norm(residual[task], 1)
            for task in range(tasks)
        ) / epsilon
        maxima["uniform_bound_excess"] = max(
            maxima["uniform_bound_excess"], gap - explicit_bound
        )

    checks = {
        "binary_delta_kernel_is_rank_one": maxima[
            "delta_kernel_identity_error"
        ] < 1e-12,
        "weighted_sum_equals_matrix_alignment": maxima[
            "weighted_alignment_error"
        ] < 1e-12,
        "exact_shared_structure_is_rank_one": maxima[
            "rank_one_residual"
        ] < 1e-12,
        "shared_direction_is_centered": maxima[
            "shared_direction_centering_error"
        ] < 1e-12,
        "projected_objective_reduces_to_direction": maxima[
            "projected_reduction_error"
        ] < 1e-12,
        "threshold_certificate_is_exact": maxima[
            "threshold_formula_error"
        ] < 1e-12,
        "threshold_value_is_separation_invariant": maxima[
            "separation_invariance_error"
        ] < 1e-12,
        "rank_one_and_full_threshold_splits_coincide": split_failures == 0,
        "global_upper_bound_holds": maxima["global_bound_excess"] < 1e-12,
        "near_optimality_transfer_holds": maxima[
            "near_optimality_excess"
        ] < 1e-12,
        "explicit_uniform_bound_holds": maxima[
            "uniform_bound_excess"
        ] < 1e-12,
    }
    result = {
        "paper_anchor": "Lemmas 6.1-6.2 and Theorem 6.3",
        "scope": (
            "Independent finite-dimensional audit of every algebraic and "
            "geometric dependency; no MIMIC data are used."
        ),
        "seeds": args.seeds,
        "max_errors_or_excesses": {
            key: float(value) for key, value in maxima.items()
        },
        "split_failures": int(split_failures),
        "checks": {key: bool(value) for key, value in checks.items()},
        "status": "passed" if all(checks.values()) else "failed",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
