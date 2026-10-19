"""Toy scalability proxy: regression versus permutation interaction estimates."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np


DIMENSIONS = (10, 12, 14, 16)
SEEDS = tuple(range(20))
NOISE_STD = 0.35


def design_row(mask: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.concatenate(
        (
            np.ones(1),
            mask.astype(float),
            np.array([mask[i] * mask[j] for i, j in pairs], dtype=float),
        )
    )


def game_value(
    mask: np.ndarray,
    main: np.ndarray,
    pair_coefficients: np.ndarray,
    pairs: list[tuple[int, int]],
) -> float:
    return float(
        main @ mask
        + sum(
            coefficient * mask[i] * mask[j]
            for coefficient, (i, j) in zip(pair_coefficients, pairs)
        )
    )


def regression_estimate(
    rng: np.random.Generator,
    dimension: int,
    budget: int,
    main: np.ndarray,
    truth: np.ndarray,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    masks = rng.integers(0, 2, size=(budget, dimension), dtype=np.int8)
    design = np.stack([design_row(mask, pairs) for mask in masks])
    response = np.array(
        [game_value(mask, main, truth, pairs) for mask in masks]
    )
    response = response + rng.normal(0.0, NOISE_STD, size=budget)
    coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
    return coefficients[1 + dimension :]


def permutation_estimate(
    rng: np.random.Generator,
    dimension: int,
    budget: int,
    main: np.ndarray,
    truth: np.ndarray,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    estimates = [[] for _ in pairs]
    evaluations = 0
    pair_order = rng.permutation(len(pairs))
    cursor = 0
    while evaluations + 4 <= budget:
        pair_index = int(pair_order[cursor % len(pair_order)])
        cursor += 1
        i, j = pairs[pair_index]
        base = rng.integers(0, 2, size=dimension, dtype=np.int8)
        base[i] = 0
        base[j] = 0
        values = []
        for include_i, include_j in ((0, 0), (1, 0), (0, 1), (1, 1)):
            mask = base.copy()
            mask[i] = include_i
            mask[j] = include_j
            noisy_value = game_value(mask, main, truth, pairs)
            noisy_value += float(rng.normal(0.0, NOISE_STD))
            values.append(noisy_value)
        estimates[pair_index].append(values[3] - values[1] - values[2] + values[0])
        evaluations += 4
    return np.array(
        [np.mean(items) if items else 0.0 for items in estimates],
        dtype=float,
    )


def rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((estimate - truth) ** 2)))


def run() -> dict:
    tasks = {}
    all_regression_wins = True
    for dimension in DIMENSIONS:
        pairs = list(combinations(range(dimension), 2))
        parameter_count = 1 + dimension + len(pairs)
        budgets = (2 * parameter_count, 4 * parameter_count)
        task_summary = {}
        for budget in budgets:
            regression_errors = []
            permutation_errors = []
            for seed in SEEDS:
                rng = np.random.default_rng(260216505 + 1000 * dimension + seed)
                main = rng.normal(0.0, 0.4, size=dimension)
                truth = rng.normal(0.0, 0.6, size=len(pairs))
                regression_errors.append(
                    rmse(
                        regression_estimate(
                            rng, dimension, budget, main, truth, pairs
                        ),
                        truth,
                    )
                )
                permutation_errors.append(
                    rmse(
                        permutation_estimate(
                            rng, dimension, budget, main, truth, pairs
                        ),
                        truth,
                    )
                )
            regression_mean = float(np.mean(regression_errors))
            permutation_mean = float(np.mean(permutation_errors))
            regression_wins = regression_mean < permutation_mean
            all_regression_wins = all_regression_wins and regression_wins
            task_summary[str(budget)] = {
                "regression_rmse_mean": regression_mean,
                "permutation_rmse_mean": permutation_mean,
                "regression_to_permutation_ratio": (
                    regression_mean / permutation_mean
                ),
                "regression_wins": regression_wins,
            }
        tasks[str(dimension)] = {
            "pair_interactions": len(pairs),
            "regression_parameters": parameter_count,
            "budgets": task_summary,
        }

    return {
        "status": "passed" if all_regression_wins else "failed",
        "paper_anchor": "Figure 7 scalability direction",
        "scope": (
            "Toy proxy only: four synthetic coalition games with 10-16 "
            "features, not the paper's four real datasets or full estimator suite."
        ),
        "seeds": len(SEEDS),
        "noise_standard_deviation": NOISE_STD,
        "tasks": tasks,
        "interpretation": (
            "At equal coalition-evaluation budgets, joint regression reuses each "
            "coalition across all pair coefficients and has lower RMSE than the "
            "pairwise permutation estimator on every synthetic task."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("results/scalability_proxy.json")
    )
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
