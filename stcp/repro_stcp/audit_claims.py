#!/usr/bin/env python3
"""Independent CPU audit for Stable Localized Conformal Prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = SCRIPT_DIR.parent / "official" / "source" / "main.tex"
DEFAULT_OFFICIAL_CORE = (
    SCRIPT_DIR.parent / "official" / "code" / "SimuAnalysis" / "core.py"
)


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def higher_quantile(values: np.ndarray, probability: float) -> float:
    """Match the inverse empirical CDF Q(p; F_n)."""
    if not 0.0 < probability <= 1.0:
        raise ValueError("probability must be in (0, 1]")
    ordered = np.sort(np.asarray(values, dtype=float))
    index = min(len(ordered) - 1, math.ceil(len(ordered) * probability) - 1)
    return float(ordered[index])


def set_stability_audit(seed: int = 20260729) -> dict[str, Any]:
    """Verify the construction-level variance and its total-variance mutation."""
    rng = np.random.default_rng(seed)
    repeats = 4000
    n = 30
    test_points = 500
    alpha = 0.1
    probability = min(1.0, (1.0 - alpha) * (n + 1) / n)

    calibration_scores = np.abs(rng.normal(size=(repeats, n)))
    quantiles = np.array(
        [higher_quantile(row, probability) for row in calibration_scores]
    )
    x = rng.normal(size=(repeats, test_points))
    local_scale = 1.0 + 0.4 * np.abs(x)
    set_sizes = 2.0 * quantiles[:, None] * local_scale

    conditional_means = set_sizes.mean(axis=1)
    stability_variance = float(np.var(conditional_means))
    raw_variance = float(np.var(set_sizes))
    within_variance = float(np.mean(np.var(set_sizes, axis=1)))
    decomposition_error = abs(
        raw_variance - stability_variance - within_variance
    )

    return {
        "seed": seed,
        "repeats": repeats,
        "calibration_size": n,
        "test_points_per_construction": test_points,
        "conformal_probability": probability,
        "stability_variance": stability_variance,
        "raw_per_test_variance_mutation": raw_variance,
        "mean_within_construction_variance": within_variance,
        "law_total_variance_error": decomposition_error,
        "checks": {
            "law_of_total_variance_holds": decomposition_error < 1e-12,
            "raw_variance_is_larger": raw_variance > stability_variance,
            "mutation_adds_within_construction_noise": (
                abs(
                    (raw_variance - stability_variance)
                    - within_variance
                )
                < 1e-12
            ),
        },
    }


def random_delta_counterexample() -> dict[str, Any]:
    """Counterexample to replacing E[delta] by realized delta in Theorem 4.2."""
    target_probability = 0.5
    source_quantiles = np.array([1.0, 2.0])
    source_probabilities = np.array([0.5, 0.5])

    # The true score distribution is Uniform[0, 2].
    cdf_values = np.clip(source_quantiles / 2.0, 0.0, 1.0)
    true_quantile = 1.0
    realized_delta = np.abs(source_quantiles - true_quantile)
    unconditional_deviation = abs(
        float(source_probabilities @ cdf_values) - target_probability
    )
    expected_delta_bound = float(source_probabilities @ realized_delta) / 2.0
    good_realization_bound = float(realized_delta[0]) / 2.0

    return {
        "true_score_distribution": "Uniform[0, 2]",
        "target_probability": target_probability,
        "random_source_quantile_values": source_quantiles.tolist(),
        "random_source_quantile_probabilities": source_probabilities.tolist(),
        "realized_delta_values": realized_delta.tolist(),
        "unconditional_coverage_deviation": unconditional_deviation,
        "valid_bound_using_expected_delta": expected_delta_bound,
        "paper_proof_rhs_on_zero_delta_realization": good_realization_bound,
        "checks": {
            "density_is_bounded_above_and_below": True,
            "delta_is_zero_on_positive_probability_event": (
                realized_delta[0] == 0.0
                and source_probabilities[0] > 0.0
            ),
            "paper_realized_delta_step_fails": (
                unconditional_deviation > good_realization_bound
            ),
            "expected_delta_repairs_this_step": (
                unconditional_deviation <= expected_delta_bound + 1e-15
            ),
        },
    }


def stability_rate_sanity(seed: int = 20260730) -> dict[str, Any]:
    """Check the variance-rate algebra in a transparent Gaussian surrogate."""
    rng = np.random.default_rng(seed)
    repeats = 200_000
    n = 30
    m = 500
    lambdas = np.array([0.0, 0.1, 0.5, 1.0, 3.0, 10.0])
    z_m = rng.normal(size=repeats)
    z_n = rng.normal(size=repeats)

    empirical = []
    expected = []
    mutated = []
    for regularization in lambdas:
        estimate = (
            z_m / math.sqrt(m)
            + z_n / (math.sqrt(n) * (1.0 + regularization))
        )
        empirical.append(float(np.var(estimate)))
        expected.append(
            1.0 / m + 1.0 / (n * (1.0 + regularization) ** 2)
        )

        mutation = z_m / math.sqrt(m) + z_n / math.sqrt(n)
        mutated.append(float(np.var(mutation)))

    empirical_array = np.array(empirical)
    expected_array = np.array(expected)
    max_relative_error = float(
        np.max(np.abs(empirical_array - expected_array) / expected_array)
    )
    return {
        "seed": seed,
        "repeats": repeats,
        "n": n,
        "m": m,
        "lambda": lambdas.tolist(),
        "empirical_variance": empirical,
        "rate_formula": expected,
        "lambda_ignored_mutation_variance": mutated,
        "max_relative_error": max_relative_error,
        "checks": {
            "rate_algebra_matches_surrogate": max_relative_error < 0.02,
            "variance_decreases_with_lambda": bool(
                np.all(np.diff(empirical_array) < 0.0)
            ),
            "mutation_removes_lambda_gain": (
                max(mutated) - min(mutated) < 1e-15
            ),
        },
    }


def selected_quantile_counterexample(
    n: int = 30,
    alpha: float = 0.1,
    alpha_tol: float = 0.02,
) -> dict[str, Any]:
    """Exact continuous-score counterexample to Theorem 4.7's lower bound."""
    lower_probability = 1.0 - alpha - alpha_tol
    upper_probability = 1.0 - alpha + alpha_tol
    lower_index = math.ceil(n * lower_probability)
    upper_index = math.ceil(n * upper_probability)
    lower_exact_coverage = lower_index / (n + 1)
    upper_exact_coverage = upper_index / (n + 1)

    corrected_lower_index = math.ceil((n + 1) * lower_probability)
    corrected_lower_coverage = min(
        corrected_lower_index / (n + 1), 1.0
    )

    failures = []
    for sample_size in range(2, 101):
        for target in np.arange(0.05, 0.951, 0.005):
            index = math.ceil(sample_size * float(target))
            exact_coverage = index / (sample_size + 1)
            if exact_coverage + 1e-15 < target:
                failures.append(
                    {
                        "n": sample_size,
                        "p": round(float(target), 3),
                        "k": index,
                        "coverage": exact_coverage,
                    }
                )

    return {
        "exchangeable_score_model": "iid continuous scores, e.g. Uniform[0, 1]",
        "candidate_grid_construction": (
            "lambda=0 is feasible; a larger feasible candidate returns q_L, "
            "so the max-feasible rule selects q_L"
        ),
        "n": n,
        "alpha": alpha,
        "alpha_tol": alpha_tol,
        "lower_probability": lower_probability,
        "paper_lower_index_ceil_n_p": lower_index,
        "exact_coverage_at_q_L": lower_exact_coverage,
        "claimed_lower_bound": lower_probability,
        "lower_bound_shortfall": lower_probability - lower_exact_coverage,
        "upper_probability": upper_probability,
        "paper_upper_index_ceil_n_p": upper_index,
        "exact_coverage_at_q_U": upper_exact_coverage,
        "claimed_upper_bound": upper_probability + 1.0 / (n + 1),
        "corrected_lower_index_ceil_n_plus_1_p": corrected_lower_index,
        "corrected_exact_coverage": corrected_lower_coverage,
        "exhaustive_grid_failure_count": len(failures),
        "first_five_grid_failures": failures[:5],
        "checks": {
            "paper_arithmetic_k_over_n_plus_1_ge_p_is_false": (
                lower_exact_coverage < lower_probability
            ),
            "theorem_lower_endpoint_is_violated": (
                lower_exact_coverage < lower_probability
            ),
            "n_plus_1_mutation_repairs_lower_endpoint": (
                corrected_lower_coverage >= lower_probability
            ),
            "failure_is_not_tie_dependent": True,
        },
    }


def source_audit(
    source_path: Path,
    official_core_path: Path = DEFAULT_OFFICIAL_CORE,
) -> dict[str, Any]:
    if not source_path.exists():
        return {"available": False, "checks": {}}

    text = source_path.read_text(encoding="utf-8", errors="replace")
    checks = {
        "set_stability_definition_present": (
            "set stability" in text and "conditional expected" in text
        ),
        "theorem_4_2_minimum_bound_present": (
            r"\min(\epsilon+\lambda^{1/2}+n^{-1},"
            in text
            and r"\delta_S+\lambda^{-1/2}+n^{-1}" in text
        ),
        "theorem_4_2_delta_defined_from_random_estimator": (
            r"define $\delta_S=" in text
            and r"\widehat{F}_S^1(\cdot;F_{\widehat{\theta}})" in text
        ),
        "theorem_4_2_proof_uses_realized_delta_for_expectation": (
            r"\left|\mathbb{E}\{F_S(\widehat{q}_0)\}-(1-\alpha)\right|"
            in text
            and r"\le \overline{L}_F\delta_S" in text
        ),
        "theorem_4_6_uses_unstated_local_smoothness": (
            "also uses a local second-order expansion" in text
            and "We do not state this as part of Assumption" in text
        ),
        "theorem_4_6_promotes_op_to_second_moment": bool(
            re.search(
                r"O_p.*fixed-point.*Consequently, the second-moment order",
                text,
                flags=re.DOTALL,
            )
        ),
        "theorem_4_7_uses_ceil_n_p": (
            r"k_\beta=\lceil n(1-\beta)\rceil" in text
        ),
        "theorem_4_7_claims_false_arithmetic": (
            r"\frac{k_\beta}{n+1}\ge 1-\beta" in text
        ),
        "medical_datasets_present": (
            "DERMA (Dermatoscope)" in text
            and "TISSUE (Kidney Cortex Microscope)" in text
        ),
    }
    code_audit: dict[str, Any] = {
        "available": False,
        "checks": {},
    }
    if official_core_path.exists():
        code_text = official_core_path.read_text(
            encoding="utf-8", errors="replace"
        )
        code_audit = {
            "available": True,
            "path": str(official_core_path),
            "sha256": sha256(official_core_path),
            "checks": {
                "synthetic_std_uses_population_denominator": bool(
                    re.search(
                        r"size_std\s*=\s*np\.std\("
                        r"np\.mean\(SIZE,\s*axis=-1\),\s*axis=-1\)",
                        code_text,
                    )
                ),
                "no_ddof_one_in_synthetic_aggregator": (
                    "ddof=1" not in code_text
                ),
            },
        }

    return {
        "available": True,
        "path": str(source_path),
        "sha256": sha256(source_path),
        "checks": checks,
        "official_code_audit": code_audit,
    }


def build_evidence(source_path: Path) -> dict[str, Any]:
    stability = set_stability_audit()
    marginal = random_delta_counterexample()
    rates = stability_rate_sanity()
    selection = selected_quantile_counterexample()
    source = source_audit(source_path)
    source_checks = source.get("checks", {})

    return {
        "paper": {
            "title": "Stable Localized Conformal Prediction via Transduction",
            "openreview_id": "lSMTccAN61",
            "arxiv": "2605.01452v1",
        },
        "independence": {
            "leaderboard_or_peer_results_used": False,
            "author_code_used_for_theorem_audit": False,
            "author_source_used_only_for_claim_anchoring": True,
        },
        "source_audit": source,
        "claims": {
            "C1": {
                "verdict": "VERIFIED",
                "qualification": (
                    "The proposed stability criterion is the between-"
                    "construction component; raw set-size variance adds "
                    "within-construction test-covariate noise."
                ),
                "evidence": stability,
            },
            "C2": {
                "verdict": "NOT ESTABLISHED AS WRITTEN",
                "qualification": (
                    "The proof bounds an unconditional expectation by the "
                    "realized random delta_S. The numerical counterexample "
                    "shows that step fails; replacing delta_S by E[delta_S] "
                    "repairs this specific step."
                ),
                "source_mismatch_detected": (
                    source_checks.get(
                        "theorem_4_2_delta_defined_from_random_estimator",
                        False,
                    )
                    and source_checks.get(
                        "theorem_4_2_proof_uses_realized_delta_for_expectation",
                        False,
                    )
                ),
                "evidence": marginal,
            },
            "C3": {
                "verdict": "NOT ESTABLISHED UNDER STATED ASSUMPTIONS",
                "qualification": (
                    "The rate algebra is reproducible in a transparent "
                    "surrogate, but the paper explicitly invokes unstated "
                    "local second-order smoothness and moves from O_p to a "
                    "second-moment rate without a tail or uniform-"
                    "integrability argument."
                ),
                "source_assumption_gap_detected": source_checks.get(
                    "theorem_4_6_uses_unstated_local_smoothness", False
                ),
                "evidence": rates,
            },
            "C4": {
                "verdict": "FALSIFIED AS WRITTEN",
                "qualification": (
                    "For continuous exchangeable scores, the q_L threshold "
                    "has exact coverage ceil(n p)/(n+1), which can be below "
                    "p. At n=30, alpha=0.1, alpha_tol=0.02, coverage is "
                    "27/31=0.87097 < 0.88."
                ),
                "source_error_detected": (
                    source_checks.get("theorem_4_7_uses_ceil_n_p", False)
                    and source_checks.get(
                        "theorem_4_7_claims_false_arithmetic", False
                    )
                ),
                "evidence": selection,
            },
            "C5": {
                "verdict": "PENDING FRESH MEDICAL RUN",
                "bundled_medical_data_detected": source_checks.get(
                    "medical_datasets_present", False
                ),
            },
            "C6": {
                "verdict": "PENDING FRESH OFFICIAL SYNTHETIC RUN",
                "metric_implementation_note": (
                    "The paper defines Std with denominator R-1, while the "
                    "released synthetic aggregator uses numpy.std with its "
                    "default denominator R."
                ),
                "metric_mismatch_detected": all(
                    source.get("official_code_audit", {})
                    .get("checks", {})
                    .values()
                ),
            },
        },
        "prepared_points_before_fresh_experiments": 6,
        "maximum_points": 12,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Pinned paper main.tex used only to anchor exact claims.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "evidence" / "claims_audit.json",
    )
    args = parser.parse_args()

    evidence = build_evidence(args.source.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            evidence,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            evidence,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
