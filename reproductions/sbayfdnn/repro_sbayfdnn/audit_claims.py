#!/usr/bin/env python3
"""Independent source, formula, and theorem-contract audit for sBayFDNN."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np


CODE_COMMIT = "276d87949c8b973bf8b6748aa8df4f56086e063e"
SOURCE_SHA256 = "f9dae2ec3769e93f5887eca672e0d94c04fcf41b50580c0dc37a070ddeef2ccf"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def independent_pip(
    norm2: np.ndarray, inclusion: float, spike_var: float, slab_var: float, width: int
) -> np.ndarray:
    log_odds = (
        math.log(inclusion / (1.0 - inclusion))
        + 0.5 * width * math.log(spike_var / slab_var)
        + 0.5 * norm2 * (1.0 / spike_var - 1.0 / slab_var)
    )
    return 1.0 / (1.0 + np.exp(-log_odds))


def independent_intervals(
    selected: np.ndarray, projection_num: int, degree: int
) -> list[tuple[float, float]]:
    inner = projection_num - degree + 1
    knots = np.concatenate(
        [np.zeros(degree), np.linspace(0.0, 1.0, inner), np.ones(degree)]
    )
    raw = sorted(
        (float(knots[j]), float(knots[j + degree + 1]))
        for j in map(int, selected)
        if knots[j + degree + 1] > knots[j]
    )
    merged: list[list[float]] = []
    for left, right in raw:
        if not merged or left > merged[-1][1] + 1e-12:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    return [(left, right) for left, right in merged]


def require_patterns(text: str, patterns: dict[str, str]) -> dict[str, bool]:
    found = {
        name: re.search(pattern, text, flags=re.DOTALL) is not None
        for name, pattern in patterns.items()
    }
    missing = [name for name, present in found.items() if not present]
    if missing:
        raise AssertionError(f"Missing paper anchors: {missing}")
    return found


def audit(code_root: Path, paper_source: Path) -> dict:
    train_path = code_root / "train_module.py"
    evaluate_path = code_root / "evaluate.py"
    paper_text = paper_source.read_text(encoding="utf-8")
    train = load_module(train_path, "sbayfdnn_train")
    evaluate = load_module(evaluate_path, "sbayfdnn_evaluate")

    inclusion = 1e-5
    spike_var = 1e-5
    slab_var = 2e-3
    width = 64
    norm2 = np.linspace(0.0, 0.012, 1201)
    expected_pip = independent_pip(
        norm2, inclusion, spike_var, slab_var, width
    )
    released_pip = train.compute_plugin_pip_from_column_norm2(
        norm2,
        lambda_n=inclusion,
        prior_sigma_0=spike_var,
        prior_sigma_1=slab_var,
        l1_width=width,
    )
    pip_error = float(np.max(np.abs(expected_pip - released_pip)))
    constants = train._compute_threshold_constants(
        inclusion, spike_var, slab_var, width
    )
    threshold = float(constants["threshold"])
    threshold_pip = float(
        independent_pip(
            np.asarray([threshold]), inclusion, spike_var, slab_var, width
        )[0]
    )

    selected = np.asarray([0, 1, 2, 17, 18, 19])
    released_intervals = evaluate.hat_intervals_from_selected_basis(
        selected, 24, 4
    )
    expected_intervals = independent_intervals(selected, 24, 4)
    interval_error = float(
        np.max(np.abs(np.asarray(released_intervals) - np.asarray(expected_intervals)))
    )

    theorem_54_anchors = require_patterns(
        paper_text,
        {
            "theorem_54": r"label\{theorem_1\}.*?H_n\^\{-2\\alpha_1\}.*?J_n\^\{-\\alpha_\\beta\\alpha_1\}",
            "only_assumptions_1_to_3": r"Suppose that Assumptions~\\ref\{assumption_design_boundedness\}--\\ref\{assumption_Holder_link\} hold",
            "bounded_network_class": r"mathcal\{NN\}_\{J_n\}.*?parameter bound.*?E_n",
            "relu_approximation": r"lem:yarotsky-1d.*?H_n\^\{-2\\alpha_1\}",
            "spline_projection": r"prop:block1-main.*?J_n\^\{-\\alpha_\\beta\\alpha_1\}",
            "proof_decomposition": r"Term I is bounded directly.*?[Tt]erm II is controlled",
        },
    )
    exponent_errors = []
    monotone_failures = 0
    for alpha_beta in (0.5, 1.0, 2.0, 3.0):
        for alpha_g in (0.4, 1.0, 1.7):
            alpha1 = min(alpha_g, 1.0)
            previous = math.inf
            for scale in (8, 16, 32, 64, 128):
                spline_then_holder = (scale ** (-alpha_beta)) ** alpha1
                composed = scale ** (-alpha_beta * alpha1)
                exponent_errors.append(abs(spline_then_holder - composed))
                bound = scale ** (-2.0 * alpha1) + composed
                if bound >= previous:
                    monotone_failures += 1
                previous = bound
    counterexample_n = np.asarray(
        [100, 400, 1_600, 6_400, 25_600, 102_400], dtype=float
    )
    counterexample_scales = np.sqrt(counterexample_n)
    positive_parameter_bound = 1.0 / counterexample_n
    output_upper_bound = 2.0 * positive_parameter_bound
    uniform_error_lower_bound = 1.0 - output_upper_bound
    claimed_rate_without_constant = (
        counterexample_scales ** -2 + counterexample_scales ** -1
    )
    counterexample_ratio = (
        uniform_error_lower_bound / claimed_rate_without_constant
    )

    theorem_57_anchors = require_patterns(
        paper_text,
        {
            "statement_upper_direction": r"varepsilon_n\^2\s*\\;\\lesssim\\;\s*\\frac\{s_n",
            "proof_required_lower_direction": r"b'n\\varepsilon_n\^2",
            "entropy_obligation": r"verifying condition \(c\)",
        },
    )
    ns = np.asarray([10, 100, 1_000, 10_000, 100_000, 1_000_000], dtype=float)
    complexity = 1.0 / ns
    epsilon2 = complexity**2
    statement_holds = epsilon2 <= complexity
    proof_ratio = complexity / epsilon2

    theorem_59_anchors = require_patterns(
        paper_text,
        {
            "posterior_q": r"Denote \$q_j=\\Pi\(r_j=1\|D_n\)",
            "map_transition": r"based on Theorem 2\.3 stated in Sun et al\..*?appropriate choice of prior hyperparameters",
            "stated_assumptions": r"Suppose Assumptions~\\ref\{assumption_design_boundedness\}--\\ref\{assumption_identifiability\} hold",
        },
    )

    missing_real_data_pipeline = all(
        not any(code_root.glob(pattern))
        for pattern in (
            "*ecg*.py",
            "*tecator*.py",
            "*real*data*.py",
            "*baseline*.py",
        )
    )

    if pip_error > 1e-12 or abs(threshold_pip - 0.5) > 1e-12:
        raise AssertionError("Released PIP formula differs from independent Bayes formula")
    if interval_error > 1e-12:
        raise AssertionError("Released interval mapping differs from independent mapping")
    if max(exponent_errors) > 1e-15 or monotone_failures:
        raise AssertionError("Approximation exponent composition audit failed")
    if not bool(np.all(statement_holds)) or proof_ratio[-1] < 1e5:
        raise AssertionError("Theorem 5.7 direction counterexample failed")

    return {
        "provenance": {
            "official_code_commit": CODE_COMMIT,
            "train_module_sha256": sha256(train_path),
            "evaluate_sha256": sha256(evaluate_path),
            "paper_tex_sha256": sha256(paper_source),
            "arxiv_source_archive_sha256": SOURCE_SHA256,
        },
        "claims": [
            {
                "id": "C1",
                "verdict": "VERIFIED",
                "score": 2,
                "pip_grid_points": int(norm2.size),
                "max_pip_absolute_error": pip_error,
                "pip_at_analytic_threshold": threshold_pip,
                "analytic_norm2_threshold": threshold,
                "interval_mapping_max_error": interval_error,
                "selected_basis_probe": selected.tolist(),
                "mapped_intervals": released_intervals,
            },
            {
                "id": "C2",
                "verdict": "FALSIFIED_AS_STATED",
                "score": 2,
                "paper_anchors": theorem_54_anchors,
                "exponent_combinations_checked": len(exponent_errors),
                "max_exponent_composition_error": max(exponent_errors),
                "monotone_bound_failures": monotone_failures,
                "counterexample": {
                    "sample_size_n": counterexample_n.astype(int).tolist(),
                    "parameter_bound_E_n": positive_parameter_bound.tolist(),
                    "hidden_width": 1,
                    "X_t": 1,
                    "beta_t": 1,
                    "g_u": "u",
                    "true_mean": 1,
                    "all_network_outputs_max_abs": output_upper_bound.tolist(),
                    "uniform_error_lower_bound": uniform_error_lower_bound.tolist(),
                    "scales_J_equals_H": counterexample_scales.astype(int).tolist(),
                    "claimed_rate_without_constant": claimed_rate_without_constant.tolist(),
                    "error_lower_bound_over_rate": counterexample_ratio.tolist(),
                },
                "finding": (
                    "Theorem 5.4 does not lower-bound E_n. With positive "
                    "E_n=1/n and width one, every admissible network output "
                    "is O(1/n), while a constant nonzero truth satisfies the "
                    "three stated assumptions."
                ),
            },
            {
                "id": "C3",
                "verdict": "INCONCLUSIVE_PROOF_GAP",
                "score": 0,
                "paper_anchors": theorem_57_anchors,
                "counterexample_n": ns.astype(int).tolist(),
                "statement_epsilon2_le_complexity": statement_holds.tolist(),
                "proof_required_complexity_over_epsilon2": proof_ratio.tolist(),
                "finding": (
                    "The statement permits epsilon^2=complexity^2, while the "
                    "proof requires complexity=O(epsilon^2); the ratio "
                    "diverges. This is a proof gap, not a failed posterior "
                    "contraction sequence."
                ),
            },
            {
                "id": "C4",
                "verdict": "INCONCLUSIVE",
                "score": 0,
                "paper_anchors": theorem_59_anchors,
                "finding": (
                    "The proof introduces posterior-to-MAP equivalence under "
                    "an appropriate hyperparameter choice not stated among "
                    "Theorem 5.9 assumptions."
                ),
            },
            {
                "id": "C5",
                "verdict": "NOT_EXECUTED",
                "score": 0,
                "released_real_data_pipeline_missing": missing_real_data_pipeline,
            },
            {
                "id": "C6",
                "verdict": "NOT_EXECUTED",
                "score": 0,
                "released_real_data_pipeline_missing": missing_real_data_pipeline,
            },
        ],
        "prepared_score": 4,
        "maximum_score": 12,
        "leaderboard_or_peer_evidence_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--paper-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.code_root, args.paper_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
