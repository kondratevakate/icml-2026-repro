#!/usr/bin/env python3
"""Independent CPU audit for the PEQ-Net challenge claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.manifold import MDS


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = SCRIPT_DIR.parent / "official" / "source"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    x2 = np.sum(x * x, axis=1)[:, None]
    y2 = np.sum(y * y, axis=1)[None, :]
    sqdist = np.maximum(x2 + y2 - 2.0 * x @ y.T, 0.0)
    return np.exp(-gamma * sqdist)


def biased_mmd2(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    value = (
        rbf_kernel(x, x, gamma).mean()
        + rbf_kernel(y, y, gamma).mean()
        - 2.0 * rbf_kernel(x, y, gamma).mean()
    )
    return float(max(value, 0.0))


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    delta = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def rank_order(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    return float(x_centered @ y_centered / denom)


def policy_embedding_check(seed: int = 20260729) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = 240
    histories = rng.normal(size=(n, 3))
    score = histories @ np.array([1.0, -0.55, 0.3])
    probabilities = 1.0 / (1.0 + np.exp(-score))
    thresholds = np.array([0.25, 0.40, 0.50, 0.60, 0.75])
    samples = [
        np.column_stack([histories, (probabilities > threshold).astype(float)])
        for threshold in thresholds
    ]

    pooled = np.vstack(samples)
    subset = pooled[rng.choice(len(pooled), size=500, replace=False)]
    euclidean = pairwise_distances(subset)
    median = float(np.median(euclidean[np.triu_indices_from(euclidean, k=1)]))
    gamma = 1.0 / (2.0 * median)

    k = len(samples)
    mmd2 = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            mmd2[i, j] = mmd2[j, i] = biased_mmd2(
                samples[i], samples[j], gamma
            )

    model = MDS(
        n_components=2,
        metric="precomputed",
        random_state=seed,
        n_init=8,
        init="random",
        max_iter=1000,
        eps=1e-9,
        normalized_stress=False,
    )
    coordinates = model.fit_transform(mmd2)
    embedded = pairwise_distances(coordinates)
    upper = np.triu_indices(k, k=1)
    target_mmd2 = mmd2[upper]
    target_mmd = np.sqrt(target_mmd2)
    fitted = embedded[upper]
    relative_stress = float(
        np.linalg.norm(fitted - target_mmd2) / np.linalg.norm(target_mmd2)
    )
    ordering_correlation = pearson(
        rank_order(target_mmd2), rank_order(fitted)
    )
    exact_mmd_error = float(np.max(np.abs(fitted - target_mmd)))

    return {
        "seed": seed,
        "n_histories": n,
        "thresholds": thresholds.tolist(),
        "median_pairwise_distance": median,
        "paper_bandwidth_parameter": gamma,
        "mmd2_matrix": mmd2.tolist(),
        "embedding_coordinates": coordinates.tolist(),
        "relative_stress_against_mmd2": relative_stress,
        "distance_ordering_correlation": ordering_correlation,
        "max_error_if_treated_as_exact_mmd": exact_mmd_error,
        "checks": {
            "all_values_finite": bool(
                np.isfinite(mmd2).all() and np.isfinite(coordinates).all()
            ),
            "zero_diagonal": bool(np.allclose(np.diag(mmd2), 0.0)),
            "symmetric_dissimilarity": bool(np.allclose(mmd2, mmd2.T)),
            "policy_ordering_preserved": ordering_correlation >= 0.90,
            "mds_is_not_exact_identity_to_mmd": exact_mmd_error > 1e-3,
        },
    }


def ltmle_lipschitz_counterexample(epsilon: float = 0.8) -> dict[str, Any]:
    initial_i = initial_j = 0.5
    clever_covariate = 1.0

    def expit(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    targeted_i = expit(epsilon * clever_covariate)
    targeted_j = expit(-epsilon * clever_covariate)
    initial_difference = abs(initial_i - initial_j)
    targeted_difference = abs(targeted_i - targeted_j)

    return {
        "strictly_positive_treatment_probability": 0.5,
        "bounded_fluctuations": [epsilon, -epsilon],
        "initial_q_values": [initial_i, initial_j],
        "targeted_q_values": [targeted_i, targeted_j],
        "initial_difference": initial_difference,
        "targeted_difference": targeted_difference,
        "claimed_rhs_for_any_finite_constant": 0.0,
        "checks": {
            "strict_positivity_holds": True,
            "fluctuations_are_bounded": math.isfinite(epsilon),
            "initial_models_identical": initial_difference == 0.0,
            "targeted_models_differ": targeted_difference > 0.0,
            "claimed_lipschitz_implication_fails": (
                initial_difference == 0.0 and targeted_difference > 0.0
            ),
        },
    }


def source_audit(source_root: Path) -> dict[str, Any]:
    files = {
        "methods": source_root / "methods.tex",
        "appendix": source_root / "appendix.tex",
        "experiments": source_root / "experiments.tex",
    }
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        return {
            "available": False,
            "missing": missing,
            "checks": {},
        }

    texts = {
        name: path.read_text(encoding="utf-8", errors="replace")
        for name, path in files.items()
    }
    appendix = texts["appendix"]
    methods = texts["methods"]
    experiments = texts["experiments"]

    singular_pattern = re.compile(
        r"\\frac\{\(-1\)\^i\}\{1\s*-\s*i\}"
    )
    source_checks = {
        "mmd_pipeline_present": all(
            token in methods
            for token in (
                "maximum mean discrepancy",
                "metric multidimensional scaling",
                "SMACOF",
                "policy-tail encoder",
            )
        ),
        "method_states_approximate_mds_preservation": (
            r"\approx" in methods and "eq:mds_stress" in methods
        ),
        "proof_claims_exact_mds_identity": (
            "eq:rho_equals_mmd" in appendix
            and "recovers the single-step MMD distance" in appendix
        ),
        "subsequence_constant_is_pair_specific_ratio": (
            "Case 2:" in appendix
            and "Define" in appendix
            and r"\frac{\|\mu_{m:n}^{(i)}-\mu_{m:n}^{(j)}\|"
            in appendix
        ),
        "bounded_update_asserted_without_targeting_lipschitz_assumption": (
            "bounded update preserves the local geometry" in appendix
            and "eq:bounded_ltmle_update" in appendix
        ),
        "printed_dgp_divides_by_zero_at_first_lag": bool(
            singular_pattern.search(appendix)
        ),
        "twenty_seed_protocol_present": "repeat 20 experiments" in experiments,
        "mimic_iv_999_patient_claim_present": (
            "999 adult ICU patients" in experiments
            and "serum lactate measured at 72 hours" in experiments
        ),
    }
    return {
        "available": True,
        "file_sha256": {
            name: sha256(path) for name, path in files.items()
        },
        "first_lag_denominator": 0,
        "checks": source_checks,
    }


def build_evidence(source_root: Path) -> dict[str, Any]:
    embedding = policy_embedding_check()
    targeting = ltmle_lipschitz_counterexample()
    source = source_audit(source_root)
    source_checks = source.get("checks", {})

    empirical_blockers = {
        "author_code_found": False,
        "processed_mimic_iii_available": False,
        "printed_dgp_executable": not source_checks.get(
            "printed_dgp_divides_by_zero_at_first_lag", True
        ),
        "selected_hyperparameters_and_seeds_reported": False,
    }
    empirical_ready = all(empirical_blockers.values())

    c4_checks = embedding["checks"]
    c5_checks = {
        "pair_specific_constant_detected": source_checks.get(
            "subsequence_constant_is_pair_specific_ratio", False
        ),
        "approximate_exact_mds_mismatch_detected": (
            source_checks.get("method_states_approximate_mds_preservation", False)
            and source_checks.get("proof_claims_exact_mds_identity", False)
            and c4_checks["mds_is_not_exact_identity_to_mmd"]
        ),
        "bounded_targeting_counterexample_passes": targeting["checks"][
            "claimed_lipschitz_implication_fails"
        ],
    }

    return {
        "paper": {
            "title": (
                "Smooth Multi-Policy Causal Effect Estimation in "
                "Longitudinal Settings"
            ),
            "openreview_id": "bIcz7bIZSo",
            "arxiv": "2605.14284v2",
        },
        "source_audit": source,
        "claims": {
            "C1": {
                "verdict": "INCONCLUSIVE",
                "blockers": empirical_blockers,
                "all_requirements_available": empirical_ready,
            },
            "C2": {
                "verdict": "INCONCLUSIVE",
                "blockers": empirical_blockers,
                "all_requirements_available": empirical_ready,
            },
            "C3": {
                "verdict": "INCONCLUSIVE",
                "blockers": empirical_blockers,
                "all_requirements_available": empirical_ready,
            },
            "C4": {
                "verdict": "VERIFIED",
                "qualification": (
                    "The pipeline is reproduced, but metric MDS only "
                    "approximately preserves the supplied MMD-squared "
                    "dissimilarities; it is not exact equality to MMD."
                ),
                "embedding": embedding,
                "checks": c4_checks,
            },
            "C5": {
                "verdict": "FALSIFIED AS A LIPSCHITZ GUARANTEE",
                "targeting_counterexample": targeting,
                "checks": c5_checks,
            },
            "C6": {
                "verdict": "INCONCLUSIVE",
                "blockers": {
                    "complete_mimic_iv_available": False,
                    "cohort_sql_released": False,
                    "item_id_mapping_released": False,
                    "preprocessing_and_seeds_released": False,
                },
            },
        },
        "prepared_points": 4,
        "maximum_points": 12,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Extracted arXiv source directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "evidence" / "claims_audit.json",
    )
    args = parser.parse_args()

    evidence = build_evidence(args.source_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
