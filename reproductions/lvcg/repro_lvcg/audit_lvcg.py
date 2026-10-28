#!/usr/bin/env python3
"""Independent release audit for LVCG (arXiv:2605.31249)."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
OFFICIAL = CANDIDATE / "official"
REPO = OFFICIAL / "repo"

EXPECTED_PAPER_SHA256 = (
    "199e996c786e10cffe8b39eca2ef26dfae5f2474573c63b3a3a9f3c556f3cdea"
)
EXPECTED_SOURCE_SHA256 = (
    "09577a0fe30f6fe5f76b65e6defb30169e146bfdb3082fe146c2832ffc4e74af"
)
EXPECTED_COMMIT = "0fcacbf34784cd876b4c197599253a9130707f93"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args],
        text=True,
        encoding="utf-8",
    ).strip()


def artifact_audit() -> dict[str, Any]:
    paper = OFFICIAL / "paper.pdf"
    source = OFFICIAL / "source.tar.gz"
    tracked = git("ls-files").splitlines()
    checkpoint_suffixes = (".pt", ".pth", ".ckpt")
    result_files = [
        name
        for name in tracked
        if name.startswith("probing/results/") and not name.endswith(".gitkeep")
    ]
    return {
        "paper": {
            "bytes": paper.stat().st_size,
            "sha256": sha256(paper),
            "expected_sha256": EXPECTED_PAPER_SHA256,
            "matches": sha256(paper) == EXPECTED_PAPER_SHA256,
        },
        "source": {
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
            "expected_sha256": EXPECTED_SOURCE_SHA256,
            "matches": sha256(source) == EXPECTED_SOURCE_SHA256,
        },
        "repository": {
            "commit": git("rev-parse", "HEAD"),
            "expected_commit": EXPECTED_COMMIT,
            "commit_matches": git("rev-parse", "HEAD") == EXPECTED_COMMIT,
            "tracked_files": len(tracked),
            "checkpoints": [
                name for name in tracked if name.lower().endswith(checkpoint_suffixes)
            ],
            "result_files": result_files,
            "test_files": [
                name
                for name in tracked
                if "test" in Path(name).name.lower()
                and name.lower().endswith(".py")
            ],
        },
    }


def missing_relative_imports() -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for path in sorted(REPO.rglob("*.py")):
        if any(part in {".git", "graphify-out"} for part in path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        relative = path.relative_to(REPO).with_suffix("")
        package = list(relative.parts[:-1])
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level == 0:
                continue
            base = package[: max(0, len(package) - node.level + 1)]
            target = base + ((node.module or "").split(".") if node.module else [])
            resolved = REPO.joinpath(*target)
            if not (
                resolved.with_suffix(".py").is_file()
                or (resolved / "__init__.py").is_file()
            ):
                missing.append(
                    {
                        "file": str(relative.with_suffix(".py")).replace("\\", "/"),
                        "line": node.lineno,
                        "import": "." * node.level + (node.module or ""),
                        "resolved": str(resolved.relative_to(REPO)).replace("\\", "/"),
                    }
                )
    return missing


def runtime_import_audit() -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import lvcg.models.vcg; from lvcg.models.lvcg import LVCG",
        ],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    stderr = proc.stderr.strip()
    return {
        "command": "import lvcg.models.vcg; from lvcg.models.lvcg import LVCG",
        "returncode": proc.returncode,
        "succeeded": proc.returncode == 0,
        "last_error_line": stderr.splitlines()[-1] if stderr else "",
        "missing_relative_imports": missing_relative_imports(),
    }


def geometry_check(seed: int = 260531249) -> dict[str, Any]:
    """Check the paper's ridge lift and fixed projection independently."""
    rng = np.random.default_rng(seed)
    batch, leads, time = 4, 5, 128
    directions = rng.normal(size=(batch, leads, 3))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    latent = rng.normal(size=(batch, 3, time))
    visible = directions @ latent
    eps = 1e-6
    recovered = np.empty_like(latent)
    for index in range(batch):
        u = directions[index]
        recovered[index] = np.linalg.solve(
            u.T @ u + eps * np.eye(3), u.T @ visible[index]
        )
    reprojection = directions @ recovered
    return {
        "seed": seed,
        "shape": {
            "directions": list(directions.shape),
            "visible": list(visible.shape),
            "latent": list(latent.shape),
        },
        "max_latent_abs_error": float(np.max(np.abs(recovered - latent))),
        "max_visible_reprojection_abs_error": float(
            np.max(np.abs(reprojection - visible))
        ),
        "passes_1e_4": bool(
            np.max(np.abs(recovered - latent)) < 1e-4
            and np.max(np.abs(reprojection - visible)) < 1e-4
        ),
    }


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]], bool]:
    raw = path.read_text(encoding="utf-8-sig")
    leading_blank = raw.startswith(("\n", "\r"))
    nonblank = (line for line in raw.splitlines() if line.strip())
    reader = csv.DictReader(nonblank)
    return list(reader.fieldnames or []), list(reader), leading_blank


def split_audit() -> dict[str, Any]:
    root = REPO / "probing" / "data_splits"
    tasks = {
        "chapman": root / "chapman",
        "icbeb": root / "icbeb",
        "ptbxl_form": root / "ptbxl" / "form",
        "ptbxl_rhythm": root / "ptbxl" / "rhythm",
        "ptbxl_sub_class": root / "ptbxl" / "sub_class",
        "ptbxl_super_class": root / "ptbxl" / "super_class",
    }
    results: dict[str, Any] = {}
    for task, directory in tasks.items():
        prefix = "ptbxl_" + task.removeprefix("ptbxl_") if task.startswith("ptbxl") else task
        tables: dict[str, tuple[list[str], list[dict[str, str]], bool]] = {}
        for split in ("train", "val", "test"):
            tables[split] = read_csv(directory / f"{prefix}_{split}.csv")
        first_columns = tables["train"][0]
        record_key = "ecg_path" if "ecg_path" in first_columns else "ecg_id"
        if record_key not in first_columns:
            record_key = "filename"
        record_sets = {
            split: {row[record_key] for row in table[1]} for split, table in tables.items()
        }
        overlaps = {
            "train_val": len(record_sets["train"] & record_sets["val"]),
            "train_test": len(record_sets["train"] & record_sets["test"]),
            "val_test": len(record_sets["val"] & record_sets["test"]),
        }
        patient_overlaps: dict[str, int] = {}
        if "patient_id" in first_columns:
            patient_sets = {
                split: {row["patient_id"] for row in table[1]}
                for split, table in tables.items()
            }
            patient_overlaps = {
                "train_val": len(patient_sets["train"] & patient_sets["val"]),
                "train_test": len(patient_sets["train"] & patient_sets["test"]),
                "val_test": len(patient_sets["val"] & patient_sets["test"]),
            }
        results[task] = {
            "rows": {split: len(table[1]) for split, table in tables.items()},
            "record_key": record_key,
            "record_overlap": overlaps,
            "patient_overlap": patient_overlaps,
            "duplicate_records": {
                split: len(table[1]) - len(record_sets[split])
                for split, table in tables.items()
            },
            "column_schema_equal": all(
                table[0] == first_columns for table in tables.values()
            ),
            "label_columns": {
                split: table[0][6:] if task.startswith("ptbxl") else table[0]
                for split, table in tables.items()
            },
            "leading_blank_line": {
                split: table[2] for split, table in tables.items()
            },
        }
    return {
        "tasks": results,
        "all_record_splits_disjoint": all(
            all(value == 0 for value in result["record_overlap"].values())
            for result in results.values()
        ),
        "all_available_patient_splits_disjoint": all(
            all(value == 0 for value in result["patient_overlap"].values())
            for result in results.values()
            if result["patient_overlap"]
        ),
    }


def table_arithmetic() -> dict[str, Any]:
    values = np.array(
        [
            [75.33, 79.03, 80.13],
            [70.61, 74.62, 79.19],
            [52.28, 59.12, 71.24],
            [72.03, 79.87, 83.94],
            [71.09, 79.44, 84.15],
            [62.47, 75.17, 84.14],
        ]
    )
    printed = np.array([67.30, 74.54, 80.47])
    calculated = values.mean(axis=0)
    return {
        "calculated_average": calculated.tolist(),
        "printed_average": printed.tolist(),
        "absolute_difference": np.abs(calculated - printed).tolist(),
        # The last mean is 80.465. The table uses conventional half-up
        # presentation (80.47), while NumPy uses ties-to-even.
        "agrees_at_printed_precision": bool(
            np.all(np.abs(calculated - printed) <= 0.0050000001)
        ),
        "cpsc_1pct_gain_over_heartlang_points": 71.09 - 60.44,
        "csn_1pct_gain_over_heartlang_points": 62.47 - 57.94,
    }


def conformance_audit() -> dict[str, Any]:
    train = (REPO / "scripts" / "train.py").read_text(encoding="utf-8")
    config = (REPO / "configs" / "train" / "lvcg_v5_gru.yaml").read_text(
        encoding="utf-8"
    )
    evaluate = (REPO / "scripts" / "evaluate.py").read_text(encoding="utf-8")
    probing = (REPO / "configs" / "eval" / "probing.yaml").read_text(
        encoding="utf-8"
    )
    paper_method = (
        OFFICIAL / "source" / "sections" / "method.tex"
    ).read_text(encoding="utf-8")
    paper_experiments = (
        OFFICIAL / "source" / "sections" / "exp.tex"
    ).read_text(encoding="utf-8")
    return {
        "paper_objective_has_three_terms": all(
            token in paper_method
            for token in ("mathcal{L}_{\\text{ECG}}", "mathcal{L}_{\\text{VCG}}", "mathcal{L}_{\\text{temp}}")
        ),
        "released_training_adds_base_loss": all(
            token in train for token in ("base_beat_loss", "lambda_base * loss_base")
        ) and "lambda_base: 1.0" in config,
        "post_paper_gru_shape_fix": {
            "commit": EXPECTED_COMMIT,
            "message": git("show", "-s", "--format=%s", EXPECTED_COMMIT),
            "changed_training_file": "scripts/train.py" in git(
                "show", "--format=", "--name-only", EXPECTED_COMMIT
            ).splitlines(),
        },
        "checkpoint_override_is_not_serialized": (
            'model_cfg["checkpoint"] = args.checkpoint' in evaluate
            and "subprocess.check_call" in evaluate
            and "yaml.safe_dump" not in evaluate
        ),
        "paper_noncardiac_dataset": "MIMIC-IV-ECG-Ext-ICD",
        "released_noncardiac_dataset": "AI-READI",
        "release_config_mentions_aireadi": "aireadi_condition" in probing,
        "release_config_mentions_mimic_ext_icd": "mimic-iv-ecg-ext-icd" in probing.lower(),
        "paper_source_mentions_mimic_ext_icd": (
            "MIMIC-IV-ECG-Ext-ICD" in paper_experiments
            or "MIMIC-IV-ECG-Ext-ICD" in (
                OFFICIAL / "paper.txt"
            ).read_text(encoding="utf-8", errors="replace")
        ),
    }


def run() -> dict[str, Any]:
    artifacts = artifact_audit()
    runtime = runtime_import_audit()
    geometry = geometry_check()
    splits = split_audit()
    arithmetic = table_arithmetic()
    conformance = conformance_audit()
    claims = [
        {"id": "C1", "score": 1, "max_score": 2},
        {"id": "C2", "score": 0, "max_score": 2},
        {"id": "C3", "score": 1, "max_score": 2},
        {"id": "C4", "score": 0, "max_score": 2},
        {"id": "C5", "score": 0, "max_score": 2},
        {"id": "C6", "score": 0, "max_score": 2},
    ]
    return {
        "paper": {
            "title": "Learning Cardiac Latent Representations in Vectorcardiogram Space",
            "openreview_id": "hS6iw4PM8K",
            "arxiv_id": "2605.31249v1",
        },
        "artifacts": artifacts,
        "runtime_import": runtime,
        "geometry": geometry,
        "splits": splits,
        "table_arithmetic": arithmetic,
        "conformance": conformance,
        "claims": claims,
        "prepared_score": sum(claim["score"] for claim in claims),
        "score_denominator": sum(claim["max_score"] for claim in claims),
        "limits": {
            "full_pretraining_run": False,
            "headline_tables_regenerated": False,
            "reason": (
                "The official release omits required internal data modules, "
                "the pretrained checkpoint, empirical outputs, and ablation artifacts."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "evidence" / "audit.json",
    )
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
