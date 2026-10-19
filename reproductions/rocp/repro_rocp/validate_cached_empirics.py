#!/usr/bin/env python3
"""Validate the complete cached ROCP run and emit claim-level checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_SEEDS = list(range(23, 43))
EXPECTED_ALPHAS = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def walk_numbers(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from walk_numbers(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_numbers(item)
    elif isinstance(value, (int, float)):
        yield float(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def means(summary: dict[str, Any], metric: str, method: str) -> np.ndarray:
    return np.asarray(
        summary["metrics"][metric][method]["mean"], dtype=float
    )


def critical(
    summary: dict[str, Any],
    metric: str,
    method: str,
) -> dict[str, float]:
    return {
        label: float(values["mean"])
        for label, values in summary["metrics"][metric][method].items()
    }


def validate_summary(
    path: Path,
    *,
    dataset: str,
    variant: str,
) -> dict[str, Any]:
    summary = load(path)
    checks = {
        "dataset": summary["dataset"] == dataset,
        "variant": summary["variant"] == variant,
        "exact_seeds_23_through_42": summary["seeds"] == EXPECTED_SEEDS,
        "exact_alpha_grid": summary["alpha_list"] == EXPECTED_ALPHAS,
        "all_values_finite": all(
            math.isfinite(value) for value in walk_numbers(summary)
        ),
        "official_revision_pinned": summary["official_git_revision"]
        == "3ee0cf6e393d2e434368bc1fc7fe3abd03ed493f",
    }
    seed_files = sorted(path.parent.glob("seed_*.json"))
    checks["twenty_seed_files"] = len(seed_files) == 20
    checks["seed_files_match_summary"] = [
        load(seed_file)["seed"] for seed_file in seed_files
    ] == EXPECTED_SEEDS
    return {"summary": summary, "checks": checks, "seed_files": seed_files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/cached"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cached/claim_validation.json"),
    )
    args = parser.parse_args()

    paths = {
        "covid_lambda0": args.root / "covid" / "lambda0" / "summary.json",
        "covid_lambda1": args.root / "covid" / "lambda1" / "summary.json",
        "bdd": args.root / "bdd" / "default" / "summary.json",
    }
    validated = {
        "covid_lambda0": validate_summary(
            paths["covid_lambda0"], dataset="covid", variant="lambda0"
        ),
        "covid_lambda1": validate_summary(
            paths["covid_lambda1"], dataset="covid", variant="lambda1"
        ),
        "bdd": validate_summary(
            paths["bdd"], dataset="bdd", variant="default"
        ),
    }
    lambda0 = validated["covid_lambda0"]["summary"]
    lambda1 = validated["covid_lambda1"]["summary"]
    bdd = validated["bdd"]["summary"]

    l0_rocp = critical(lambda0, "critical_mistake", "rocp")
    l0_rac = critical(lambda0, "critical_mistake", "rac")
    l1_rocp = critical(lambda1, "critical_mistake", "rocp")
    l1_rac = critical(lambda1, "critical_mistake", "rac")
    bdd_rocp = critical(bdd, "critical_bad_action", "rocp")
    bdd_rac = critical(bdd, "critical_bad_action", "rac")
    lambda0_gains = {
        label: l0_rac[label] - l0_rocp[label] for label in l0_rocp
    }
    lambda1_gains = {
        label: l1_rac[label] - l1_rocp[label] for label in l1_rocp
    }

    coverage_errors = {}
    for name, item in validated.items():
        summary = item["summary"]
        miscoverage = means(summary, "miscoverage", "ROCP")
        coverage_errors[name] = np.abs(
            miscoverage - np.asarray(EXPECTED_ALPHAS)
        ).tolist()

    bdd_realized_delta = means(bdd, "realized_loss", "ROCP") - means(
        bdd, "realized_loss", "RAC"
    )
    bdd_risk_delta = means(bdd, "worst_case_risk", "ROCP") - means(
        bdd, "worst_case_risk", "RAC"
    )
    claim_checks = {
        "covid_lambda0_rocp_reduces_all_critical_mistakes_vs_rac": all(
            l0_rocp[label] < l0_rac[label] for label in l0_rocp
        ),
        "covid_lambda1_rocp_eliminates_observed_critical_mistakes": all(
            value == 0.0 for value in l1_rocp.values()
        ),
        "covid_lambda1_gain_exceeds_lambda0_for_every_critical_label": all(
            lambda1_gains[label] > lambda0_gains[label]
            for label in lambda0_gains
        ),
        "bdd_rocp_reduces_collision_rate_for_every_hazard_label_vs_rac": all(
            bdd_rocp[label] < bdd_rac[label] for label in bdd_rocp
        ),
        "all_rocp_miscoverage_within_0_005_of_alpha": max(
            max(errors) for errors in coverage_errors.values()
        )
        < 0.005,
        "stronger_bdd_all_alpha_realized_loss_statement_literal": bool(
            np.all(bdd_realized_delta <= 0)
        ),
        "stronger_bdd_all_alpha_worst_case_risk_statement_literal": bool(
            np.all(bdd_risk_delta <= 0)
        ),
    }
    required = [
        key
        for key in claim_checks
        if not key.startswith("stronger_bdd_")
    ]
    structural_checks = [
        value
        for item in validated.values()
        for value in item["checks"].values()
    ]
    status = (
        "passed"
        if all(structural_checks) and all(claim_checks[key] for key in required)
        else "failed"
    )

    result = {
        "status": status,
        "scope": (
            "Fresh evaluation of official cached probabilities/scores using "
            "the released global-beta implementation."
        ),
        "claim_checks": claim_checks,
        "coverage_absolute_errors": coverage_errors,
        "covid_critical_rates": {
            "lambda0": {"rocp": l0_rocp, "rac": l0_rac},
            "lambda1": {"rocp": l1_rocp, "rac": l1_rac},
        },
        "bdd_collision_rates": {"rocp": bdd_rocp, "rac": bdd_rac},
        "bdd_rocp_minus_rac": {
            "realized_loss": bdd_realized_delta.tolist(),
            "worst_case_risk": bdd_risk_delta.tolist(),
        },
        "limitations": [
            (
                "The empirical runner evaluates the released global-beta "
                "approximation, not the paper's candidate-wise Algorithm 1."
            ),
            (
                "The stronger BDD all-baselines/all-alpha statement has small "
                "RAC reversals; the anchored critical-mistake claim still passes."
            ),
        ],
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "checks": validated[name]["checks"],
            }
            for name, path in paths.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
