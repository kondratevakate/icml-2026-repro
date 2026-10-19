#!/usr/bin/env python3
"""Independent static and released-dataset audit for MA-RAG."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
CODE = CANDIDATE / "official" / "code"
DATASETS = CODE / "datasets"
TEX = CANDIDATE / "official" / "arxiv-source" / "ma_rag_cr.tex"
EVIDENCE = HERE / "evidence" / "artifact_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_question(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def audit_dataset(path: Path) -> tuple[dict[str, Any], dict[str, set[str]]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    ids: list[Any] = []
    exact_questions: list[str] = []
    normalized_questions: list[str] = []
    option_counts: Counter[int] = Counter()
    effective_option_counts: Counter[int] = Counter()
    answer_counts: Counter[str] = Counter()
    blank_option_rows = 0
    multi_letter_answer_count = 0
    question_targets: dict[str, set[str]] = {}

    if not isinstance(records, list):
        raise ValueError(f"{path.name}: root must be a list")

    for row_index, row in enumerate(records):
        if not isinstance(row, dict):
            errors.append(f"row {row_index}: not an object")
            continue
        for field in ("id", "question", "option", "answer"):
            if field not in row:
                errors.append(f"row {row_index}: missing {field}")
        if errors and len(errors) > 100:
            break
        if not all(field in row for field in ("id", "question", "option", "answer")):
            continue

        question = row["question"]
        options = row["option"]
        answer = row["answer"]
        if not isinstance(question, str) or not question.strip():
            errors.append(f"row {row_index}: empty/non-string question")
        answer_keys: list[str] = []
        if isinstance(options, dict):
            if answer in options:
                answer_keys = [answer]
            elif (
                isinstance(answer, str)
                and len(answer) > 1
                and all(key in options for key in answer)
            ):
                answer_keys = list(answer)

        if not isinstance(options, dict) or len(options) < 2:
            errors.append(f"row {row_index}: fewer than two options")
        elif not answer_keys:
            errors.append(f"row {row_index}: answer not in options")
        else:
            nonblank_options = {
                key: value
                for key, value in options.items()
                if isinstance(value, str) and value.strip()
            }
            if len(nonblank_options) < 2:
                errors.append(f"row {row_index}: fewer than two nonblank options")
            if len(nonblank_options) < len(options):
                blank_option_rows += 1
            if any(
                not isinstance(options[key], str) or not options[key].strip()
                for key in answer_keys
            ):
                errors.append(f"row {row_index}: answer points to blank option")
            effective_option_counts[len(nonblank_options)] += 1

        ids.append(row["id"])
        exact_questions.append(question)
        normalized = normalize_question(question)
        normalized_questions.append(normalized)
        option_counts[len(options) if isinstance(options, dict) else 0] += 1
        answer_counts[str(answer)] += 1
        if isinstance(answer, str) and len(answer) > 1:
            multi_letter_answer_count += 1
        if isinstance(options, dict) and answer_keys:
            target = " || ".join(str(options[key]) for key in answer_keys)
            question_targets.setdefault(normalized, set()).add(target)

    duplicate_ids = len(ids) - len({json.dumps(item, sort_keys=True) for item in ids})
    duplicate_exact = len(exact_questions) - len(set(exact_questions))
    duplicate_normalized = len(normalized_questions) - len(set(normalized_questions))
    result = {
        "file": path.name,
        "sha256": sha256(path),
        "rows": len(records),
        "unique_ids": len(ids) - duplicate_ids,
        "duplicate_ids": duplicate_ids,
        "duplicate_exact_questions": duplicate_exact,
        "duplicate_normalized_questions": duplicate_normalized,
        "conflicting_normalized_question_groups": sum(
            len(targets) > 1 for targets in question_targets.values()
        ),
        "option_count_distribution": dict(sorted(option_counts.items())),
        "effective_option_count_distribution": dict(
            sorted(effective_option_counts.items())
        ),
        "blank_option_rows": blank_option_rows,
        "multi_letter_answer_count": multi_letter_answer_count,
        "answer_distribution": dict(sorted(answer_counts.items())),
        "errors": errors,
        "passed": not errors and duplicate_ids == 0,
    }
    sets = {
        "exact": set(exact_questions),
        "normalized": set(normalized_questions),
    }
    return result, sets


def audit_all_datasets(dataset_dir: Path = DATASETS) -> dict[str, Any]:
    results: dict[str, Any] = {}
    question_sets: dict[str, dict[str, set[str]]] = {}
    for path in sorted(dataset_dir.glob("*.json")):
        result, sets = audit_dataset(path)
        results[path.stem] = result
        question_sets[path.stem] = sets

    overlaps: dict[str, dict[str, int]] = {}
    names = sorted(question_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            exact = len(question_sets[left]["exact"] & question_sets[right]["exact"])
            normalized = len(
                question_sets[left]["normalized"] & question_sets[right]["normalized"]
            )
            if exact or normalized:
                overlaps[f"{left}::{right}"] = {
                    "exact": exact,
                    "normalized": normalized,
                }

    runner_incompatible_rows = sum(
        item["multi_letter_answer_count"] for item in results.values()
    )
    return {
        "dataset_count": len(results),
        "total_rows": sum(item["rows"] for item in results.values()),
        "all_pass": len(results) == 7 and all(item["passed"] for item in results.values()),
        "single_answer_runner_incompatible_rows": runner_incompatible_rows,
        "single_answer_runner_compatible": runner_incompatible_rows == 0,
        "datasets": results,
        "cross_dataset_question_overlaps": overlaps,
    }


def audit_architecture(code_dir: Path = CODE, tex_path: Path = TEX) -> dict[str, Any]:
    entropy_path = code_dir / "ma_rag_entropy.py"
    evaluator_path = code_dir / "ma_rag_evaluator.py"
    utils_path = code_dir / "utils.py"
    entropy = entropy_path.read_text(encoding="utf-8")
    evaluator = evaluator_path.read_text(encoding="utf-8")
    utils = utils_path.read_text(encoding="utf-8")
    tex = tex_path.read_text(encoding="utf-8")

    checks = {
        "round_loop_entropy": "range(1, local_args.num_round + 1)" in entropy,
        "round_loop_evaluator": "range(1, local_args.num_round + 1)" in evaluator,
        "consensus_early_stop_entropy": (
            "len(set([round_result['prediction'] for round_result in round_results])) == 1"
            in entropy
        ),
        "consensus_early_stop_evaluator": (
            "len(set([round_result['prediction'] for round_result in round_results])) == 1"
            in evaluator
        ),
        "conflict_query_generation_entropy": (
            "queries = re.findall" in entropy and "Query .*?" in entropy
        ),
        "conflict_query_generation_evaluator": (
            "queries = re.findall" in evaluator and "Query .*?" in evaluator
        ),
        "retrieval_top32_top2_entropy": (
            "total_k=32, top_k=2" in entropy and "use_reranker=True" in entropy
        ),
        "retrieval_top32_top2_evaluator": (
            "total_k=32, top_k=2" in evaluator and "use_reranker=True" in evaluator
        ),
        "majority_vote_accuracy": "Counter([result['prediction']" in utils,
        "paper_requires_descending_quality": (
            "quality scores follow a descending order" in tex
        ),
        "paper_intrinsic_score_is_negative_entropy": (
            "Q_{\\text{int}}(a) = -\\frac{1}{L}" in tex
        ),
        "paper_describes_epsilon_gate": (
            "confidence threshold $\\epsilon$ as a gating mechanism" in tex
        ),
        "released_uses_unanimity_gate": (
            "len(set([round_result['prediction'] for round_result in round_results])) == 1"
            in entropy
            and "len(set([round_result['prediction'] for round_result in round_results])) == 1"
            in evaluator
        ),
        "released_has_configurable_epsilon": (
            "--epsilon" in entropy or "--epsilon" in evaluator
        ),
        "released_entropy_orders_high_entropy_first": (
            "np.argsort(previous_answer_entropies)[::-1]" in entropy
        ),
        "released_evaluator_orders_low_score_first": (
            "arg_scores = np.argsort(scores)" in evaluator
            and "previous_answer_contents[i] for i in arg_scores" in evaluator
        ),
    }
    core_checks = [
        key
        for key in checks
        if key.startswith(("round_", "consensus_", "conflict_", "retrieval_"))
    ]
    ranking_direction_matches_paper = not (
        checks["paper_requires_descending_quality"]
        and checks["paper_intrinsic_score_is_negative_entropy"]
        and checks["released_entropy_orders_high_entropy_first"]
        and checks["released_evaluator_orders_low_score_first"]
    )
    return {
        "checks": checks,
        "core_loop_checks_pass": all(checks[key] for key in core_checks),
        "ranking_direction_matches_paper": ranking_direction_matches_paper,
        "ranking_direction_finding": (
            "Both released ranking branches order candidates opposite to the "
            "paper's descending-quality specification."
            if not ranking_direction_matches_paper
            else "No ranking-direction mismatch detected."
        ),
        "entropy_sha256": sha256(entropy_path),
        "evaluator_sha256": sha256(evaluator_path),
        "utils_sha256": sha256(utils_path),
    }


def paper_table_arithmetic() -> dict[str, Any]:
    qwen = [71.1, 61.3, 51.0, 64.9, 56.0, 67.2, 16.1]
    ma_rag_ext = [77.1, 67.2, 59.1, 70.7, 60.5, 78.4, 22.2]
    tc_rag = [70.0, 64.6, 49.0, 63.6, 57.7, 75.2, 18.0]
    multi_refine = [74.7, 63.6, 55.8, 69.1, 58.0, 73.6, 16.3]

    mean = lambda values: sum(values) / len(values)
    return {
        "qwen_mean_unrounded": mean(qwen),
        "ma_rag_ext_mean_unrounded": mean(ma_rag_ext),
        "tc_rag_mean_unrounded": mean(tc_rag),
        "multi_refine_mean_unrounded": mean(multi_refine),
        "ma_rag_minus_qwen_unrounded": mean(ma_rag_ext) - mean(qwen),
        "ma_rag_minus_tc_rag_unrounded": mean(ma_rag_ext) - mean(tc_rag),
        "ma_rag_minus_multi_refine_unrounded": (
            mean(ma_rag_ext) - mean(multi_refine)
        ),
        "note": "Source-table arithmetic only; not experimental reproduction.",
    }


def build_evidence() -> dict[str, Any]:
    datasets = audit_all_datasets()
    architecture = audit_architecture()
    score_c1 = (
        2
        if architecture["core_loop_checks_pass"]
        and architecture["ranking_direction_matches_paper"]
        else 1 if architecture["core_loop_checks_pass"] else 0
    )
    score_c2 = (
        2
        if datasets["all_pass"] and datasets["single_answer_runner_compatible"]
        else 1 if datasets["all_pass"] else 0
    )
    prepared_score = score_c1 + score_c2
    return {
        "audit_version": 1,
        "architecture": architecture,
        "released_datasets": datasets,
        "paper_table_arithmetic": paper_table_arithmetic(),
        "claims": {
            "C1": {
                "points": score_c1,
                "verdict": "PARTIAL_CODE_PRESENT_RANKING_DIRECTION_MISMATCH",
            },
            "C2": {
                "points": score_c2,
                "verdict": "PARTIAL_SEVEN_DATASETS_PRESENT_RUNNER_INCOMPATIBILITY",
            },
            "C3": {"points": 0, "verdict": "INCONCLUSIVE_NOT_RERUN"},
            "C4": {"points": 0, "verdict": "INCONCLUSIVE_NOT_RERUN"},
            "C5": {"points": 0, "verdict": "INCONCLUSIVE_NOT_RERUN"},
            "C6": {"points": 0, "verdict": "INCONCLUSIVE_NOT_RERUN"},
        },
        "prepared_score": prepared_score,
        "score_denominator": 12,
        "passed": datasets["all_pass"] and architecture["core_loop_checks_pass"],
    }


def main() -> int:
    evidence = build_evidence()
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
