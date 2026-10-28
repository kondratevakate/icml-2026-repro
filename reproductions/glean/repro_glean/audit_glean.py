from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
OFFICIAL = CANDIDATE / "official"

EXPECTED_PAPER_SHA256 = (
    "1ee9bf721494de9ef78cb06ef396583de90cef8bb8093b18e6d8b9c71e291ff6"
)
EXPECTED_SOURCE_SHA256 = (
    "ceb733675f7da5f1c3991f47d5b77506009e02d584aed2b3541074f8bf2d4016"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_inventory(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        columns = next(reader)
        rows = sum(1 for _ in reader)
    return {"rows": rows, "columns": columns, "bytes": path.stat().st_size}


def official_checksum_audit(root: Path) -> dict[str, Any]:
    manifest = root / "SHA256SUMS.txt"
    checked = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        path = root / name
        actual = sha256(path)
        checked.append(
            {
                "file": name,
                "expected": expected,
                "actual": actual,
                "matches": expected.lower() == actual.lower(),
            }
        )
    return {
        "files_checked": len(checked),
        "all_match": all(item["matches"] for item in checked),
        "files": checked,
    }


def clinical_dataset_audit(root: Path) -> dict[str, Any]:
    checksums = official_checksum_audit(root)
    pathology = json.loads((root / "pathology_ids.json").read_text(encoding="utf-8"))
    counts = {name: len(ids) for name, ids in sorted(pathology.items())}
    inventories = {
        path.name: csv_inventory(path) for path in sorted(root.glob("*.csv"))
    }
    radiology_path = root / "radiology_reports.csv"
    with radiology_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    note_ids = [row["note_id"] for row in rows]
    return {
        "official_checksums": checksums,
        "pathology_counts": counts,
        "pathology_total": sum(counts.values()),
        "glean_diseases_present": all(
            name in counts
            for name in ("diverticulitis", "cholecystitis", "pancreatitis")
        ),
        "csv_inventory": inventories,
        "radiology": {
            "rows": len(rows),
            "unique_note_ids": len(set(note_ids)),
            "empty_text_rows": sum(not row["text"].strip() for row in rows),
            "official_page_claim": 5959,
            "difference_from_official_page": len(rows) - 5959,
        },
    }


def gzip_header(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle))


def notes_availability(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {"available": False}
    results: dict[str, Any] = {"available": root.is_dir(), "tables": {}}
    if not root.is_dir():
        return results
    for name in ("discharge.csv.gz", "radiology.csv.gz"):
        path = root / name
        results["tables"][name] = {
            "available": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "columns": gzip_header(path) if path.is_file() else [],
        }
    return results


def accumulation_check() -> dict[str, Any]:
    scores = [0.60, 0.80, 0.70]
    beta = 0.5
    logits = [math.log(score / (1.0 - score)) for score in scores]
    direct = sum(beta ** (len(scores) - 1 - index) * value for index, value in enumerate(logits))
    recurrent = 0.0
    states = []
    for value in logits:
        recurrent = beta * recurrent + value
        states.append(recurrent)
    return {
        "scores": scores,
        "beta": beta,
        "direct": direct,
        "recurrent": recurrent,
        "absolute_difference": abs(direct - recurrent),
        "states": states,
    }


def proof_audit() -> dict[str, Any]:
    # Appendix A assumes E[(p_S-p*)^2] <= epsilon_suff. The triangle step
    # therefore contributes at most 2*epsilon_suff, not 2*epsilon_suff**2.
    epsilon = 0.2
    supported_term = 2.0 * epsilon
    printed_term = 2.0 * epsilon**2
    return {
        "assumption_term": "E[(p_S(S)-p*(tau))^2] <= epsilon_suff",
        "supported_sufficiency_term": "2 * epsilon_suff",
        "printed_sufficiency_term": "2 * epsilon_suff^2",
        "counterexample_epsilon": epsilon,
        "supported_numeric_upper_term": supported_term,
        "printed_numeric_term": printed_term,
        "printed_term_follows_from_assumption": printed_term >= supported_term,
    }


def headline_arithmetic() -> dict[str, Any]:
    active_auc = [0.9756, 0.9063, 0.9325, 0.9862, 0.9582, 0.9061]
    active_brier = [0.0586, 0.1124, 0.1031, 0.0453, 0.0753, 0.1194]
    self_consistency_auc = [0.9011, 0.7978, 0.7621, 0.9131, 0.8493, 0.8232]
    self_consistency_brier = [0.1677, 0.1484, 0.2446, 0.1608, 0.1534, 0.2096]
    cell_best_auc = [0.9011, 0.8190, 0.8461, 0.9131, 0.8493, 0.8590]
    cell_best_brier = [0.1677, 0.1484, 0.2298, 0.1608, 0.1534, 0.1841]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    def ratio_of_means_gain(new: list[float], old: list[float]) -> float:
        return (mean(new) - mean(old)) / mean(old)

    def ratio_of_means_reduction(new: list[float], old: list[float]) -> float:
        return (mean(old) - mean(new)) / mean(old)

    return {
        "versus_self_consistency": {
            "auroc_relative_gain": ratio_of_means_gain(
                active_auc, self_consistency_auc
            ),
            "brier_relative_reduction": ratio_of_means_reduction(
                active_brier, self_consistency_brier
            ),
        },
        "versus_best_baseline_per_cell": {
            "mean_relative_auroc_gain": sum(
                (new - old) / old for new, old in zip(active_auc, cell_best_auc)
            )
            / len(active_auc),
            "mean_relative_brier_reduction": sum(
                (old - new) / old
                for new, old in zip(active_brier, cell_best_brier)
            )
            / len(active_brier),
        },
    }


def prose_table_check() -> dict[str, Any]:
    comparisons = [
        ("K3 AUROC", 0.9789, 0.9794),
        ("K3 Risk@0.5", 0.0802, 0.0741),
        ("K3 Brier", 0.0647, 0.0632),
        ("Active AUROC", 0.9856, 0.9862),
        ("Active Risk@0.5", 0.0494, 0.0370),
    ]
    rows = [
        {
            "metric": metric,
            "main_results_prose": prose,
            "table_1": table,
            "absolute_difference": abs(prose - table),
        }
        for metric, prose, table in comparisons
    ]
    return {
        "context": "Qwen3-30B diverticulitis example",
        "all_values_conflict": all(row["absolute_difference"] > 0 for row in rows),
        "comparisons": rows,
    }


def guideline_metadata() -> dict[str, Any]:
    size_path = OFFICIAL / "guidelines-size.json"
    split_path = OFFICIAL / "guidelines-splits.json"
    validity_path = OFFICIAL / "guidelines-is-valid.json"
    return {
        "size": json.loads(size_path.read_text(encoding="utf-8")),
        "splits": json.loads(split_path.read_text(encoding="utf-8")),
        "validity": json.loads(validity_path.read_text(encoding="utf-8")),
    }


def artifact_audit() -> dict[str, Any]:
    paper = OFFICIAL / "paper.pdf"
    source = OFFICIAL / "source.tar.gz"
    source_files = [path for path in (OFFICIAL / "source").rglob("*") if path.is_file()]
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
            "extracted_file_count": len(source_files),
        },
        "official_glean_code_found": False,
        "released_trajectory_or_prediction_files_found": False,
    }


def run(clinical_root: Path, notes_root: Path | None) -> dict[str, Any]:
    result = {
        "paper": {
            "title": "Guideline-Grounded Evidence Accumulation for High-Stakes Agent Verification",
            "openreview_id": "FP23eFYhAy",
            "arxiv_id": "2603.02798v1",
        },
        "artifacts": artifact_audit(),
        "guideline_dataset": guideline_metadata(),
        "clinical_dataset": clinical_dataset_audit(clinical_root),
        "mimic_iv_note": notes_availability(notes_root),
        "accumulation_check": accumulation_check(),
        "proof_audit": proof_audit(),
        "headline_arithmetic": headline_arithmetic(),
        "prose_table_consistency": prose_table_check(),
        "claim_scores": {"C1": 1, "C2": 1, "C3": 0, "C4": 0, "C5": 0, "C6": 0},
        "prepared_score": 2,
        "score_denominator": 12,
        "limits": [
            "No official GLEAN implementation or checkpoint was released.",
            "No paper-specific trajectories, labels, ratings, predictions, or seeds were released.",
            "No active-verification traces or Best-of-N candidate groups were released.",
            "No clinician-level study annotations or analysis code were released.",
        ],
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clinical-root",
        type=Path,
        default=CANDIDATE.parent / "data" / "mimic-iv-ext-cdm-1.1",
    )
    parser.add_argument(
        "--notes-root",
        type=Path,
        default=CANDIDATE.parent / "data" / "mimic-iv-note-2.2" / "release" / "note",
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "evidence" / "audit.json"
    )
    args = parser.parse_args()
    result = run(args.clinical_root.resolve(), args.notes_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "prepared_score": result["prepared_score"],
                "clinical_checksums_match": result["clinical_dataset"][
                    "official_checksums"
                ]["all_match"],
                "pathology_total": result["clinical_dataset"]["pathology_total"],
                "proof_bound_valid_as_written": result["proof_audit"][
                    "printed_term_follows_from_assumption"
                ],
                "table_prose_consistent": not result["prose_table_consistency"][
                    "all_values_conflict"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

