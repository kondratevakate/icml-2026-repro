"""Exact SurvFD functional decomposition on all three prediction scales."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np


FEATURES = 3
GRID_VALUES = np.linspace(-1.0, 1.0, 5)
TIMES = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
BASELINE_HAZARD = 0.08
SUBSETS = [
    subset
    for size in range(FEATURES + 1)
    for subset in combinations(range(FEATURES), size)
]


def grid() -> np.ndarray:
    mesh = np.meshgrid(*([GRID_VALUES] * FEATURES), indexing="ij")
    return np.stack([part.reshape(-1) for part in mesh], axis=1)


def risk_parts(x: np.ndarray, scenario: str) -> tuple[np.ndarray, np.ndarray]:
    additive = 0.6 * x[:, 0] - 0.4 * x[:, 1] + 0.3 * x[:, 2]
    if scenario == "additive_time_independent":
        time_exponent = np.zeros(len(x))
    elif scenario == "time_dependent_pair":
        time_exponent = 0.5 * x[:, 0] * x[:, 1]
    else:
        raise ValueError(scenario)
    return additive, time_exponent


def target_values(x: np.ndarray, scenario: str, scale: str) -> np.ndarray:
    additive, exponent = risk_parts(x, scenario)
    values = []
    for time in TIMES:
        log_hazard = (
            np.log(BASELINE_HAZARD)
            + additive
            + exponent * np.log1p(time)
        )
        if scale == "log_hazard":
            current = log_hazard
        elif scale == "hazard":
            current = np.exp(log_hazard)
        elif scale == "survival":
            power = exponent + 1.0
            integrated_time = ((1.0 + time) ** power - 1.0) / power
            cumulative_hazard = (
                BASELINE_HAZARD * np.exp(additive) * integrated_time
            )
            current = np.exp(-cumulative_hazard)
        else:
            raise ValueError(scale)
        values.append(current)
    return np.stack(values, axis=1)


def conditional_projection(
    values: np.ndarray,
    x: np.ndarray,
    subset: tuple[int, ...],
) -> np.ndarray:
    if not subset:
        return np.broadcast_to(values.mean(axis=0), values.shape).copy()
    projection = np.empty_like(values)
    keys = x[:, subset]
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    for group_index in range(len(unique_keys)):
        members = inverse == group_index
        projection[members] = values[members].mean(axis=0)
    return projection


def functional_anova(
    values: np.ndarray,
    x: np.ndarray,
) -> dict[tuple[int, ...], np.ndarray]:
    effects: dict[tuple[int, ...], np.ndarray] = {}
    for subset in SUBSETS:
        effect = conditional_projection(values, x, subset)
        for lower_subset, lower_effect in effects.items():
            if set(lower_subset) < set(subset):
                effect = effect - lower_effect
        effects[subset] = effect
    return effects


def summarize_effects(
    values: np.ndarray,
    effects: dict[tuple[int, ...], np.ndarray],
) -> dict:
    reconstruction = sum(effects.values())
    reconstruction_error = float(np.max(np.abs(values - reconstruction)))
    subset_summaries = {}
    for subset, effect in effects.items():
        label = "empty" if not subset else ",".join(map(str, subset))
        time_spread = float(np.max(np.ptp(effect, axis=1)))
        magnitude = float(np.sqrt(np.mean(effect**2)))
        subset_summaries[label] = {
            "order": len(subset),
            "rms_magnitude": magnitude,
            "maximum_time_spread": time_spread,
            "time_dependent": time_spread > 1e-10,
            "nonzero": magnitude > 1e-10,
        }
    return {
        "maximum_reconstruction_error": reconstruction_error,
        "subsets": subset_summaries,
    }


def run() -> dict:
    x = grid()
    scenarios = {}
    for scenario in ("additive_time_independent", "time_dependent_pair"):
        scenarios[scenario] = {}
        for scale in ("log_hazard", "hazard", "survival"):
            values = target_values(x, scenario, scale)
            effects = functional_anova(values, x)
            scenarios[scenario][scale] = summarize_effects(values, effects)

    maximum_reconstruction_error = max(
        item["maximum_reconstruction_error"]
        for scenario in scenarios.values()
        for item in scenario.values()
    )
    additive_log = scenarios["additive_time_independent"]["log_hazard"]["subsets"]
    additive_hazard = scenarios["additive_time_independent"]["hazard"]["subsets"]
    additive_survival = scenarios["additive_time_independent"]["survival"]["subsets"]
    pair_log = scenarios["time_dependent_pair"]["log_hazard"]["subsets"]

    log_higher_orders_zero = all(
        not item["nonzero"]
        for item in additive_log.values()
        if item["order"] >= 2
    )
    hazard_has_higher_order = any(
        item["nonzero"]
        for item in additive_hazard.values()
        if item["order"] >= 2
    )
    survival_has_time_dependent_higher_order = any(
        item["nonzero"] and item["time_dependent"]
        for item in additive_survival.values()
        if item["order"] >= 2
    )
    explicit_pair_is_time_dependent = pair_log["0,1"]["time_dependent"]

    passed = (
        maximum_reconstruction_error < 1e-12
        and log_higher_orders_zero
        and hazard_has_higher_order
        and survival_has_time_dependent_higher_order
        and explicit_pair_is_time_dependent
    )
    return {
        "status": "passed" if passed else "failed",
        "paper_anchor": "Section 3; Theorems 3.2-3.3; Corollary 3.4; Proposition 3.5",
        "measure": "uniform counting measure on a 5^3 Cartesian feature grid",
        "feature_grid_points": len(x),
        "time_points": TIMES.tolist(),
        "maximum_reconstruction_error": maximum_reconstruction_error,
        "checks": {
            "additive_log_hazard_has_no_higher_order_effects": log_higher_orders_zero,
            "hazard_link_creates_higher_order_effects": hazard_has_higher_order,
            "survival_scale_has_time_dependent_higher_order_effects": (
                survival_has_time_dependent_higher_order
            ),
            "explicit_log_hazard_pair_is_time_dependent": explicit_pair_is_time_dependent,
        },
        "scenarios": scenarios,
        "interpretation": (
            "The exact decomposition reconstructs every target on log-hazard, "
            "hazard, and survival scales. The nonlinear links create higher-order "
            "effects even when the log-risk is additive, and an explicit "
            "time-dependent pair is correctly classified."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/three_scales.json"))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
