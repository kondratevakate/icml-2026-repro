#!/usr/bin/env python3
"""Independent LP audit of Lemma 2.1 and Theorem 2.2 (finite-label case)."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from rocp_core import robust_closed_form


def primal_adversarial_values(
    loss_matrix: np.ndarray, label_set: frozenset[int], alpha: float
) -> np.ndarray:
    """Solve definition (2) directly for each action using linear programming."""

    num_labels, num_actions = loss_matrix.shape
    coverage_row = np.asarray(
        [-1.0 if label in label_set else 0.0 for label in range(num_labels)]
    )
    values = []
    for action in range(num_actions):
        result = linprog(
            c=-loss_matrix[:, action],
            A_ub=coverage_row[None, :],
            b_ub=np.asarray([-(1.0 - alpha)]),
            A_eq=np.ones((1, num_labels)),
            b_eq=np.ones(1),
            bounds=[(0.0, None)] * num_labels,
            method="highs",
        )
        if not result.success:
            raise RuntimeError(result.message)
        values.append(float(-result.fun))
    return np.asarray(values)


def nonempty_subsets(num_labels: int) -> list[frozenset[int]]:
    return [
        frozenset(index for index in range(num_labels) if mask & (1 << index))
        for mask in range(1, 1 << num_labels)
    ]


def run_audit(random_cases: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    alphas = (0.0, 0.05, 0.2, 0.5, 1.0)
    matrices: list[np.ndarray] = []

    # Exhaustive small integer-loss family plus larger deterministic random cases.
    matrices.extend(
        np.asarray(values, dtype=float).reshape(2, 2)
        for values in product((0.0, 1.0, 4.0), repeat=4)
    )
    for _ in range(random_cases):
        num_labels = int(rng.integers(2, 6))
        num_actions = int(rng.integers(2, 5))
        matrices.append(rng.integers(0, 101, size=(num_labels, num_actions)).astype(float))

    comparisons = 0
    max_error = 0.0
    policy_tie_set_failures = 0
    for matrix in matrices:
        for label_set in nonempty_subsets(matrix.shape[0]):
            for alpha in alphas:
                formula = robust_closed_form(matrix, label_set, alpha)
                primal = primal_adversarial_values(matrix, label_set, alpha)
                error = float(np.max(np.abs(formula - primal)))
                max_error = max(max_error, error)
                if error > 1e-8:
                    raise AssertionError(
                        f"Lemma 2.1 mismatch: error={error}, set={label_set}, "
                        f"alpha={alpha}, matrix={matrix.tolist()}"
                    )
                formula_minimizers = set(
                    np.flatnonzero(formula <= np.min(formula) + 1e-9).tolist()
                )
                primal_minimizers = set(
                    np.flatnonzero(primal <= np.min(primal) + 1e-9).tolist()
                )
                if formula_minimizers != primal_minimizers:
                    policy_tie_set_failures += 1
                comparisons += matrix.shape[1]

    if policy_tie_set_failures:
        raise AssertionError(f"{policy_tie_set_failures} minimizer-set mismatches")

    counterexample_matrix = np.asarray(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [100.0, 2.0],
        ]
    )
    counterexample_set = frozenset((0, 1))
    counterexample_alpha = 0.05
    in_set_maxima = np.max(counterexample_matrix[list(counterexample_set), :], axis=0)
    robust_values = robust_closed_form(
        counterexample_matrix, counterexample_set, counterexample_alpha
    )
    max_min_action = int(np.argmin(in_set_maxima))
    robust_action = int(np.argmin(robust_values))
    if (max_min_action, robust_action) != (0, 1):
        raise AssertionError("the asymmetric-loss counterexample did not separate policies")

    return {
        "status": "passed",
        "seed": seed,
        "matrices": len(matrices),
        "action_value_comparisons": comparisons,
        "alphas": list(alphas),
        "maximum_absolute_lp_error": max_error,
        "policy_minimizer_set_failures": policy_tie_set_failures,
        "counterexample": {
            "loss_matrix_labels_by_actions": counterexample_matrix.tolist(),
            "label_set": sorted(counterexample_set),
            "alpha": counterexample_alpha,
            "in_set_maxima": in_set_maxima.tolist(),
            "robust_values": robust_values.tolist(),
            "max_min_action": max_min_action,
            "rocp_action": robust_action,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-cases", type=int, default=120)
    parser.add_argument("--seed", type=int, default=260200989)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_audit(args.random_cases, args.seed)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
