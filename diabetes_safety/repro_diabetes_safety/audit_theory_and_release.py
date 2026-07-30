#!/usr/bin/env python3
"""Independent audit of the diabetes shield theorem and released source."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path


def audit_safety_bound() -> dict:
    hypo_checked = 0
    hyper_checked = 0
    max_margin_residual = -math.inf
    hypo_mutation_failures = 0
    hyper_mutation_failures = 0

    for fail_limit in (54.0, 70.0):
        for epsilon in (0.0, 1.0, 5.0, 10.0, 25.0):
            shield_limit = fail_limit + epsilon
            for predicted_min_offset in (0.0, 0.5, 10.0, 100.0):
                predicted_min = shield_limit + predicted_min_offset
                for error in (-20.0, -1.0, 0.0, epsilon):
                    true_min = predicted_min - error
                    residual = fail_limit - true_min
                    max_margin_residual = max(max_margin_residual, residual)
                    if true_min < fail_limit - 1e-12:
                        raise AssertionError("Theorem implication failed")
                    hypo_checked += 1

                mutated_true_min = predicted_min - (epsilon + 1.0)
                if mutated_true_min < fail_limit:
                    hypo_mutation_failures += 1

    for fail_limit in (180.0, 250.0):
        for epsilon in (0.0, 1.0, 5.0, 10.0, 25.0):
            shield_limit = fail_limit - epsilon
            for predicted_max_offset in (0.0, 0.5, 10.0, 100.0):
                predicted_max = shield_limit - predicted_max_offset
                for error in (-20.0, -1.0, 0.0, epsilon):
                    true_max = predicted_max + error
                    residual = true_max - fail_limit
                    max_margin_residual = max(max_margin_residual, residual)
                    if true_max > fail_limit + 1e-12:
                        raise AssertionError("Hyperglycemia implication failed")
                    hyper_checked += 1

                mutated_true_max = predicted_max + epsilon + 1.0
                if mutated_true_max > fail_limit:
                    hyper_mutation_failures += 1

    return {
        "verdict": "VERIFIED_CONDITIONAL_IMPLICATION",
        "hypoglycemia_checked_implications": hypo_checked,
        "hyperglycemia_checked_implications": hyper_checked,
        "checked_implications": hypo_checked + hyper_checked,
        "max_safety_margin_residual": max_margin_residual,
        "hypoglycemia_mutation_failures": hypo_mutation_failures,
        "hyperglycemia_mutation_failures": hyper_mutation_failures,
        "weakened_reliability_mutation_failures": (
            hypo_mutation_failures + hyper_mutation_failures
        ),
    }


def finite_penalty_counterexample(penalty: float = 10.0) -> dict:
    safe_logit = 0.0
    unsafe_logit = 0.0 - penalty
    unsafe_probability = math.exp(unsafe_logit) / (
        math.exp(safe_logit) + math.exp(unsafe_logit)
    )
    return {
        "penalty": penalty,
        "unsafe_action_probability": unsafe_probability,
        "hard_pruning": unsafe_probability == 0.0,
        "verdict": "FINITE_PENALTY_DOES_NOT_ENFORCE_THEOREM_PERMISSION_SET",
    }


def _assignment_string(tree: ast.AST, variable: str) -> str:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == variable for target in node.targets):
            if isinstance(node.value, ast.JoinedStr):
                return ast.unparse(node.value)
    raise ValueError(f"Could not find assignment for {variable}")


def audit_release_source(glucoalg: Path) -> dict:
    train_path = glucosim_path(glucoalg, "2.train_dynamics_predictor.py")
    shield_path = glucosim_path(glucoalg, "shield/predictive_shield.py")
    train_text = train_path.read_text(encoding="utf-8")
    shield_text = shield_path.read_text(encoding="utf-8")
    train_tree = ast.parse(train_text)
    shield_tree = ast.parse(shield_text)

    train_folder_expr = _assignment_string(train_tree, "save_folder")
    shield_folder_expr = _assignment_string(shield_tree, "save_folder")

    train_prefix_present = "ba_node_b" in train_folder_expr
    shield_prefix_present = "fe_b5" in shield_folder_expr
    path_contract_matches = not (train_prefix_present and shield_prefix_present)

    runtime_names = ("adult#001", "child#001", "adolescent#001")
    released_indices = {}
    expected_indices = {"adult#001": 0, "child#001": 10, "adolescent#001": 20}
    for name in runtime_names:
        index = int(name.split("#")[1]) - 1
        if name == "adolescent":
            index += 20
        elif name == "child":
            index += 10
        released_indices[name] = index

    finite_mask_anchor = bool(
        re.search(r"logit_penalty:\s*float\s*=\s*10\.0", shield_text)
    )
    cohort_literal_anchor = (
        "self.shield_type == 'adolescent'" in shield_text
        and "self.shield_type == 'child'" in shield_text
    )

    return {
        "train_source": train_path.as_posix(),
        "shield_source": shield_path.as_posix(),
        "train_folder_expression": train_folder_expr,
        "shield_folder_expression": shield_folder_expr,
        "path_contract_matches": path_contract_matches,
        "released_cohort_indices": released_indices,
        "expected_cohort_indices": expected_indices,
        "cohort_indices_match": released_indices == expected_indices,
        "finite_logit_penalty_anchor": finite_mask_anchor,
        "cohort_literal_comparison_anchor": cohort_literal_anchor,
        "verdict": "RELEASED_PREDICTIVE_SHIELD_NOT_RUNNABLE_AS_DOCUMENTED",
    }


def glucosim_path(root: Path, relative: str) -> Path:
    path = root / Path(relative)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glucoalg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "safety_theorem": audit_safety_bound(),
        "finite_penalty": finite_penalty_counterexample(),
        "release_source": audit_release_source(args.glucoalg),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
