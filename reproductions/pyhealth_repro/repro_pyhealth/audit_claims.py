#!/usr/bin/env python3
"""Independent source audit for PyHealth 2.0 challenge claims C1, C4, and C5."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PAPER = SCRIPT_DIR.parent / "paper_source"
DEFAULT_REPO = SCRIPT_DIR.parent / "official_pyhealth"

CHALLENGE_CLAIMS = {
    "C1": (
        "PyHealth 2.0 unifies 15+ datasets, 20+ clinical tasks, 25+ models, "
        "and 5+ interpretability methods (Attention-Grad, GIM, DeepLift, SHAP "
        "among them) spanning EHR, imaging, physiological signal, and genomic "
        "modalities (Section 3; Appendices F-J)."
    ),
    "C4": (
        "PyHealth 2.0 reduces the code required to implement standard ML tasks "
        "(e.g., mortality prediction) to as few as 7 lines, down from 24 lines "
        "in PyHealth 1.16 and up to 51 lines with raw pandas (Table 2)."
    ),
    "C5": (
        "PyHealth 2.0 has an active open-source community of 400+ members and "
        "provides 50+ tutorials alongside multi-language support via an R "
        "interface (RHealth) (Section 5, Discussion)."
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
        encoding="utf-8",
    ).strip()


def table_for_label(text: str, label: str) -> str:
    marker = rf"\label{{{label}}}"
    position = text.find(marker)
    if position < 0:
        raise ValueError(f"missing LaTeX table label: {label}")
    start = text.rfind(r"\begin{table", 0, position)
    end = text.find(r"\end{table", position)
    if start < 0 or end < 0:
        raise ValueError(f"could not bound LaTeX table: {label}")
    end = text.find("}", end) + 1
    return text[start:end]


def category_counts(table: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in table.splitlines():
        match = re.match(r"^\s*([^%\\][^&]*?)\s*&\s*(\d+)\+?\s*&", line)
        if match:
            counts[match.group(1).strip()] = int(match.group(2))
    return counts


def imported_names(init_file: Path) -> list[str]:
    tree = ast.parse(init_file.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.extend(alias.asname or alias.name for alias in node.names)
    return sorted(set(names))


def exported_inventory(repo: Path) -> dict[str, list[str]]:
    package = repo / "pyhealth"
    datasets = [
        name
        for name in imported_names(package / "datasets" / "__init__.py")
        if name.endswith("Dataset")
        and not name.startswith(("Base", "Sample"))
    ]
    tasks = [
        name
        for name in imported_names(package / "tasks" / "__init__.py")
        if name != "BaseTask"
    ]
    models = [
        name
        for name in imported_names(package / "models" / "__init__.py")
        if not name.endswith(("Layer", "Block"))
        and name
        not in {
            "BaseModel",
            "DenseBlock",
            "DenseLayer",
            "ResBlock2D",
            "SinusoidalTimeEmbedding",
            "TFMTokenizer",
            "TFM_TOKEN_Classifier",
            "TFM_VQVAE2_deep",
            "get_tfm_token_classifier_64x4",
            "get_tfm_tokenizer_2x2x8",
            "load_embedding_weights",
        }
    ]
    interpreters = [
        name
        for name in imported_names(
            package / "interpret" / "methods" / "__init__.py"
        )
        if name
        not in {
            "BaseInterpreter",
            "BaseEnsemble",
            "RandomBaseline",
            "CrhEnsemble",
            "AvgEnsemble",
            "VarEnsemble",
        }
    ]
    return {
        "datasets": datasets,
        "tasks": tasks,
        "models": models,
        "interpreters": interpreters,
    }


def audit_c1(paper: Path, repo: Path) -> dict[str, Any]:
    catalog_text = (
        paper / "sections" / "appendix_datasets_models_task.tex"
    ).read_text(encoding="utf-8")
    interp_text = (
        paper / "sections" / "appendix_processors_uq_interp.tex"
    ).read_text(encoding="utf-8")

    paper_counts = {
        "datasets": category_counts(
            table_for_label(catalog_text, "tab:dataset_overview")
        ),
        "tasks": category_counts(
            table_for_label(catalog_text, "tab:task_overview")
        ),
        "models": category_counts(
            table_for_label(catalog_text, "tab:model_overview")
        ),
        "interpretability": category_counts(
            table_for_label(interp_text, "tab:interpretability_overview")
        ),
    }
    paper_totals = {
        "datasets": sum(paper_counts["datasets"].values()),
        "tasks": sum(paper_counts["tasks"].values()),
        "models": sum(paper_counts["models"].values()),
        "interpretability_methods": sum(
            value
            for key, value in paper_counts["interpretability"].items()
            if key != "Visualization Tools"
        ),
    }

    exports = exported_inventory(repo)
    modality_witnesses = {
        "EHR": "MIMIC4Dataset",
        "imaging": "ChestXray14Dataset",
        "physiological_signal": "SleepEDFDataset",
        "genomics": "ClinVarDataset",
    }
    method_witnesses = {
        "Attention-Grad": "CheferRelevance",
        "GIM": "GIM",
        "DeepLift": "DeepLift",
        "SHAP": "ShapExplainer",
    }

    checks = {
        "paper_dataset_count_at_least_15": paper_totals["datasets"] >= 15,
        "paper_task_count_at_least_20": paper_totals["tasks"] >= 20,
        "paper_model_count_at_least_25": paper_totals["models"] >= 25,
        "paper_interpretability_count_at_least_5": (
            paper_totals["interpretability_methods"] >= 5
        ),
        "repo_dataset_exports_at_least_15": len(exports["datasets"]) >= 15,
        "repo_task_exports_at_least_20": len(exports["tasks"]) >= 20,
        "repo_model_exports_at_least_25": len(exports["models"]) >= 25,
        "repo_interpreter_exports_at_least_5": len(exports["interpreters"]) >= 5,
        "all_modalities_have_exported_witnesses": all(
            witness in exports["datasets"]
            for witness in modality_witnesses.values()
        ),
        "all_named_methods_are_exported": all(
            witness in exports["interpreters"]
            for witness in method_witnesses.values()
        ),
    }
    return {
        "claim": CHALLENGE_CLAIMS["C1"],
        "verdict": "VERIFIED" if all(checks.values()) else "FALSIFIED",
        "paper_category_counts": paper_counts,
        "paper_totals": paper_totals,
        "repo_export_counts": {
            key: len(value) for key, value in exports.items()
        },
        "repo_exports": exports,
        "modality_witnesses": modality_witnesses,
        "method_witnesses": method_witnesses,
        "checks": checks,
        "scope": (
            "Full source-level inventory at the paper's public v2.0.1 release; "
            "this claim is about toolkit coverage, not benchmark performance."
        ),
    }


def parse_loc_rows(table: str) -> dict[str, list[int]]:
    rows: dict[str, list[int]] = {}
    for line in table.splitlines():
        if "&" not in line or not line.rstrip().endswith(r"\\"):
            continue
        clean = re.sub(r"\\textbf\{(\d+)\}", r"\1", line)
        clean = re.sub(r"\\textcolor\{[^}]+\}\{[^}]*\}", "", clean)
        parts = [part.strip() for part in clean[:-2].split("&")]
        if len(parts) != 5:
            continue
        values = []
        for part in parts[1:]:
            match = re.search(r"\d+", part)
            if not match:
                break
            values.append(int(match.group()))
        if len(values) == 4:
            rows[parts[0]] = values
    return rows


def audit_c4(paper: Path) -> dict[str, Any]:
    results_text = (paper / "sections" / "results.tex").read_text(
        encoding="utf-8"
    )
    abstract_text = (paper / "sections" / "abstract.tex").read_text(
        encoding="utf-8"
    )
    methodology_text = (
        paper / "sections" / "methodology.tex"
    ).read_text(encoding="utf-8")
    table = table_for_label(results_text, "tab:loc_comparison")
    rows = parse_loc_rows(table)
    columns = [
        "patient_exploration",
        "mortality_prediction",
        "length_of_stay",
        "drug_recommendation",
    ]
    structured_rows = {
        method: dict(zip(columns, values)) for method, values in rows.items()
    }

    expected = {
        "PyHealth 2.0": 7,
        "PyHealth 1.16": 24,
        "Pandas": 51,
    }
    observed = {
        method: structured_rows[method]["mortality_prediction"]
        for method in expected
    }
    checks = {
        "abstract_contains_7_line_statement": (
            "as few as 7 lines of code" in abstract_text
        ),
        "methodology_contains_7_line_statement": (
            "training in 7 lines of code" in methodology_text
        ),
        "table2_has_expected_pandas_51": observed["Pandas"] == 51,
        "table2_has_claimed_pyhealth_116_24": (
            observed["PyHealth 1.16"] == 24
        ),
        "table2_has_claimed_pyhealth_20_7": observed["PyHealth 2.0"] == 7,
    }
    composite_supported = all(checks.values())
    return {
        "claim": CHALLENGE_CLAIMS["C4"],
        "verdict": "VERIFIED" if composite_supported else "FALSIFIED",
        "expected_mortality_loc_from_challenge_claim": expected,
        "observed_table2": structured_rows,
        "observed_mortality_loc": observed,
        "checks": checks,
        "finding": (
            "The paper does make a separate seven-line training statement, "
            "but Table 2 reports mortality-task counts of 34 for PyHealth 2.0, "
            "27 for PyHealth 1.16, and 51 for Pandas. The anchored challenge "
            "claim conflates the standalone seven-line statement with Table 2 "
            "and gives two Table 2 values incorrectly."
        ),
    }


def audit_c5(paper: Path, repo: Path) -> dict[str, Any]:
    abstract = (paper / "sections" / "abstract.tex").read_text(
        encoding="utf-8"
    )
    introduction = (
        paper / "sections" / "introduction.tex"
    ).read_text(encoding="utf-8")
    discussion = (paper / "sections" / "discussion.tex").read_text(
        encoding="utf-8"
    )
    bibliography = (paper / "sections" / "bib.bib").read_text(
        encoding="utf-8"
    )
    example_files = sorted(
        str(path.relative_to(repo)).replace("\\", "/")
        for path in (repo / "examples").rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".ipynb"}
    )
    checks: dict[str, bool | None] = {
        "paper_asserts_400_plus_members": "400+ members" in abstract,
        "paper_asserts_50_plus_examples_tutorials": (
            "50+ examples and tutorials" in introduction
        ),
        "repo_has_at_least_50_example_or_tutorial_files": (
            len(example_files) >= 50
        ),
        "paper_discussion_names_rhealth": "RHealth" in discussion,
        "paper_bibliography_links_rhealth_repository": (
            "https://github.com/v1xerunt/RHealth" in bibliography
        ),
        "independent_historical_membership_artifact_available": None,
    }
    return {
        "claim": CHALLENGE_CLAIMS["C5"],
        "verdict": "INCONCLUSIVE",
        "example_or_tutorial_file_count": len(example_files),
        "checks": checks,
        "finding": (
            "The public release contains 50+ example/tutorial files and the "
            "paper identifies a public RHealth repository. The paper provides "
            "no immutable roster or measurement procedure for the historical "
            "'400+ members' component, so the composite claim cannot be fully "
            "verified or falsified independently."
        ),
    }


def build_result(paper: Path, repo: Path) -> dict[str, Any]:
    paper_files = [
        paper / "sections" / "abstract.tex",
        paper / "sections" / "methodology.tex",
        paper / "sections" / "results.tex",
        paper / "sections" / "appendix_datasets_models_task.tex",
        paper / "sections" / "appendix_processors_uq_interp.tex",
        paper / "sections" / "discussion.tex",
        paper / "sections" / "bib.bib",
    ]
    repo_files = [
        repo / "pyhealth" / "datasets" / "__init__.py",
        repo / "pyhealth" / "tasks" / "__init__.py",
        repo / "pyhealth" / "models" / "__init__.py",
        repo / "pyhealth" / "interpret" / "methods" / "__init__.py",
    ]
    missing = [str(path) for path in paper_files + repo_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required inputs:\n" + "\n".join(missing))

    return {
        "paper": {
            "title": (
                "PyHealth 2.0: A Comprehensive Open-Source Toolkit for "
                "Accessible and Reproducible Clinical Deep Learning"
            ),
            "openreview_id": "gMLVFN9hl8",
            "arxiv": "2601.16414v2",
        },
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "script_sha256": sha256(Path(__file__)),
        },
        "provenance": {
            "arxiv_source_archive_sha256": (
                sha256(paper / "source.tar")
                if (paper / "source.tar").is_file()
                else None
            ),
            "official_repo_url": "https://github.com/sunlabuiuc/PyHealth.git",
            "official_repo_tag": "v2.0.1",
            "official_repo_commit": git_output(repo, "rev-parse", "HEAD"),
            "official_repo_clean": git_output(repo, "status", "--porcelain") == "",
            "input_sha256": {
                str(path.relative_to(paper.parent)).replace("\\", "/"): sha256(path)
                for path in paper_files
            }
            | {
                str(path.relative_to(repo.parent)).replace("\\", "/"): sha256(path)
                for path in repo_files
            },
        },
        "claims": {
            "C1": audit_c1(paper, repo),
            "C4": audit_c4(paper),
            "C5": audit_c5(paper, repo),
        },
        "not_attempted": {
            "C2": (
                "Requires the full MIMIC-IV v2.2 benchmark, exact hardware, "
                "worker sweep, PyHealth 1.16, MEDS, and Pandas baselines."
            ),
            "C3": (
                "Requires full MIMIC-IV v2.2 drug-recommendation and "
                "length-of-stay throughput benchmark runs."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-source", type=Path, default=DEFAULT_PAPER)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "evidence" / "claims_audit.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_result(args.paper_source.resolve(), args.repo.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for claim_id in ("C1", "C4", "C5"):
        claim = result["claims"][claim_id]
        print(f"{claim_id}: {claim['verdict']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
