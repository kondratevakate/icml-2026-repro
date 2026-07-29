#!/usr/bin/env python3
"""Independent, network-free audit of 3DMedAgent release artifacts."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_csv(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    types = collections.Counter(row["question type"] for row in rows)
    subtype_pairs = collections.Counter(
        (row["question type"], row["question subtype"]) for row in rows
    )
    errors = []
    ids = [row["Question ID"] for row in rows]
    questions = [row["multiple-choice question"].strip() for row in rows]
    scan_questions = [
        (row["Image ID"], row["multiple-choice question"].strip()) for row in rows
    ]
    scan_question_counts = collections.Counter(scan_questions)
    scan_question_rows = collections.defaultdict(list)
    for row in rows:
        scan_question_rows[
            (row["Image ID"], row["multiple-choice question"].strip())
        ].append(row)
    conflicting_answers = [
        {
            "image_id": key[0],
            "question": key[1],
            "question_ids": [row["Question ID"] for row in group],
            "correct_options": sorted({row["correct option"] for row in group}),
        }
        for key, group in scan_question_rows.items()
        if len({row["correct option"] for row in group}) > 1
    ]
    if len(ids) != len(set(ids)):
        errors.append("duplicate Question ID")
    if any(not row["correct option"].strip() for row in rows):
        errors.append("missing correct option")
    return {
        "sha256": sha256(path),
        "row_count": len(rows),
        "unique_image_ids": len({row["Image ID"] for row in rows}),
        "unique_question_ids": len(set(ids)),
        "unique_mcq_texts": len(set(questions)),
        "unique_scan_mcq_pairs": len(set(scan_questions)),
        "repeated_scan_mcq_pair_rows": len(scan_questions)
        - len(set(scan_questions)),
        "repeated_scan_mcq_pairs": [
            {"image_id": image_id, "question": question, "count": count}
            for (image_id, question), count in scan_question_counts.items()
            if count > 1
        ],
        "conflicting_answer_group_count": len(conflicting_answers),
        "conflicting_answer_groups": conflicting_answers,
        "option_only_mcq_count": sum(
            row["multiple-choice question"].strip().startswith("A:")
            for row in rows
        ),
        "question_type_counts": dict(sorted(types.items())),
        "subtype_count": len(subtype_pairs),
        "subtype_counts": {
            f"{kind}::{subtype}": count
            for (kind, subtype), count in sorted(subtype_pairs.items())
        },
        "subtypes_per_type": dict(
            sorted(
                collections.Counter(kind for kind, _ in subtype_pairs).items()
            )
        ),
        "split_counts": dict(
            sorted(collections.Counter(row["split"] for row in rows).items())
        ),
        "dataset_counts": dict(
            sorted(collections.Counter(row["dataset"] for row in rows).items())
        ),
        "errors": errors,
        "passed": not errors,
    }


def audit_architecture(code_root: Path) -> dict:
    pipeline = code_root / "Final_Test" / "GPT_memory_t1s.py"
    memory = code_root / "Final_Test" / "memory.py"
    readme = code_root / "README.md"
    pipeline_text = pipeline.read_text(encoding="utf-8")
    memory_text = memory.read_text(encoding="utf-8")
    readme_text = readme.read_text(encoding="utf-8")
    checks = {
        "memory_build_present": "def build_facts_memory(" in memory_text,
        "all_organ_memory_present": "def build_all_organ_memory(" in memory_text,
        "clip_global_present": "def get_clip_global_memory(" in memory_text,
        "clip_detail_present": "def get_clip_detail_memory(" in memory_text,
        "clip_slice_present": "def get_clip_detail_slice_memory(" in memory_text,
        "t1s_flag_present": 'parser.add_argument("--include-t1s"' in pipeline_text,
        "t1s_limit_flag_present": 'parser.add_argument("--t1s-max-iters"' in pipeline_text,
        "t1s_loop_uses_limit": 'getattr(args, "t1s_max_iters", 1)' in pipeline_text,
        "paper_setting_documented": "--t1s-max-iters 5" in readme_text,
        "runtime_tools_flag_present": 'parser.add_argument("--include-runtime-tools"' in pipeline_text,
        "dry_run_present": 'parser.add_argument("--dry-run"' in pipeline_text,
    }
    return {
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "pipeline_sha256": sha256(pipeline),
        "memory_sha256": sha256(memory),
        "t1s_cli_default": 1,
        "paper_readme_invocation": 5,
        "note": "The paper setting is supplied explicitly; the CLI default is one iteration.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--code-root", type=Path, default=CANDIDATE / "official" / "code"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "evidence" / "artifact_audit.json",
    )
    args = parser.parse_args()
    result = {
        "audit_version": 1,
        "benchmark": audit_csv(args.code_root / "DeepChestVQA_v1.csv"),
        "architecture": audit_architecture(args.code_root),
        "claims": {
            "C1": {"verdict": "VERIFIED_RELEASED_ARCHITECTURE", "points": 2},
            "C2": {"verdict": "VERIFIED", "points": 2},
            "C3": {"verdict": "INCONCLUSIVE_NOT_RERUN", "points": 0},
            "C4": {"verdict": "INCONCLUSIVE_NOT_RERUN", "points": 0},
            "C5": {"verdict": "INCONCLUSIVE_NOT_RERUN", "points": 0},
            "C6": {"verdict": "INCONCLUSIVE_MISSING_EXPERT_TRACES", "points": 0},
        },
        "prepared_score": 4,
        "score_denominator": 12,
    }
    result["passed"] = (
        result["benchmark"]["passed"]
        and result["architecture"]["all_checks_pass"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
