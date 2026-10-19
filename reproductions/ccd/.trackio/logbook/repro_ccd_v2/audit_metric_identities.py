"""Finite-state audit of the ERT proper-score identities in Section 3.1."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


TARGET = 0.9


def l1_f(p: float) -> float:
    return abs(p - TARGET)


def l1_df(p: float) -> float:
    if p > TARGET:
        return 1.0
    if p < TARGET:
        return -1.0
    return 0.0


def l2_f(p: float) -> float:
    return (p - TARGET) ** 2


def l2_df(p: float) -> float:
    return 2.0 * (p - TARGET)


def kl_f(p: float) -> float:
    return p * math.log(p / TARGET) + (1.0 - p) * math.log(
        (1.0 - p) / (1.0 - TARGET)
    )


def kl_df(p: float) -> float:
    return math.log(p / TARGET) - math.log((1.0 - p) / (1.0 - TARGET))


def expected_supporting_score(
    true_probability: float,
    prediction: float,
    f,
    df,
) -> float:
    slope = df(prediction)
    return f(prediction) + (true_probability - prediction) * slope


def run() -> dict:
    functions = {
        "L1": (l1_f, l1_df),
        "L2": (l2_f, l2_df),
        "KL": (kl_f, kl_df),
    }
    probabilities = [i / 100.0 for i in range(1, 100)]
    candidate_grid = probabilities + [TARGET]
    summaries = {}

    for name, (f, df) in functions.items():
        maximum_identity_error = 0.0
        maximum_target_score = 0.0
        maximum_supporting_plane_violation = 0.0
        for p in probabilities:
            oracle_score = expected_supporting_score(p, p, f, df)
            maximum_identity_error = max(
                maximum_identity_error, abs(oracle_score - f(p))
            )
            target_score = expected_supporting_score(p, TARGET, f, df)
            maximum_target_score = max(maximum_target_score, abs(target_score))
            for h in candidate_grid:
                score = expected_supporting_score(p, h, f, df)
                maximum_supporting_plane_violation = max(
                    maximum_supporting_plane_violation, score - f(p)
                )

        summaries[name] = {
            "maximum_oracle_identity_error": maximum_identity_error,
            "maximum_constant_target_score": maximum_target_score,
            "maximum_supporting_plane_violation": max(
                0.0, maximum_supporting_plane_violation
            ),
        }

    tolerance = 1e-12
    passed = all(
        item["maximum_oracle_identity_error"] <= tolerance
        and item["maximum_constant_target_score"] <= tolerance
        and item["maximum_supporting_plane_violation"] <= tolerance
        for item in summaries.values()
    )
    return {
        "status": "passed" if passed else "failed",
        "paper_anchor": "Section 3.1 and Table 1",
        "target_coverage": TARGET,
        "probability_grid_size": len(probabilities),
        "prediction_grid_size": len(candidate_grid),
        "metrics": summaries,
        "interpretation": (
            "For L1, L2, and KL, the expected supporting score equals f(p) "
            "at the true conditional coverage, is zero for the constant "
            "target predictor, and never exceeds f(p)."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/metric_identities.json"))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
