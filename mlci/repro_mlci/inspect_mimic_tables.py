#!/usr/bin/env python3
"""Find MIMIC source tables and verify the columns needed by the paper."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path


TABLES = {
    "admissions": {"subject_id", "hadm_id", "admittime", "dischtime", "deathtime"},
    "patients": {"subject_id", "dod"},
    "diagnoses_icd": {"subject_id", "hadm_id", "icd_code"},
    "icustays": {"subject_id", "hadm_id", "intime"},
}
SUFFIXES = (".csv", ".csv.gz", ".parquet")


def normalized_stem(path: Path) -> str:
    name = path.name.lower()
    for suffix in (".gz", ".csv", ".parquet"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def release_from_path(path: Path) -> str:
    text = str(path).lower().replace("_", "-")
    if "mimic-iv" in text or "mimiciv" in text:
        return "mimic-iv"
    if "mimic-iii" in text or "mimiciii" in text:
        return "mimic-iii"
    return "unknown"


def read_columns(path: Path) -> list[str]:
    if path.name.lower().endswith(".parquet"):
        try:
            import pandas as pd
        except ImportError:
            return []
        return [str(column).lower() for column in pd.read_parquet(path).columns]

    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        return [column.strip().lower() for column in next(reader)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default=r"C:\Projects\data\digitaltwin"
    )
    parser.add_argument(
        "--output", default="results/mimic_readiness.json"
    )
    args = parser.parse_args()

    root = Path(args.root)
    candidates: dict[str, list[dict]] = {name: [] for name in TABLES}
    errors = []
    if root.is_dir():
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            lower_name = path.name.lower()
            if not any(lower_name.endswith(suffix) for suffix in SUFFIXES):
                continue
            table = normalized_stem(path)
            if table not in TABLES:
                continue
            try:
                columns = read_columns(path)
                missing = sorted(TABLES[table] - set(columns))
                if table == "diagnoses_icd" and release_from_path(path) == "mimic-iv":
                    if "icd_version" not in columns:
                        missing.append("icd_version")
                candidates[table].append(
                    {
                        "path": str(path.resolve()),
                        "release": release_from_path(path),
                        "bytes": path.stat().st_size,
                        "columns": columns,
                        "missing_required_columns": sorted(set(missing)),
                    }
                )
            except Exception as error:
                errors.append({"path": str(path), "error": repr(error)})

    release_status = {}
    for release in ("mimic-iv", "mimic-iii"):
        ready_tables = []
        missing_tables = []
        for table, rows in candidates.items():
            valid = [
                row
                for row in rows
                if row["release"] == release
                and not row["missing_required_columns"]
            ]
            if valid:
                ready_tables.append(table)
            else:
                missing_tables.append(table)
        release_status[release] = {
            "ready_tables": ready_tables,
            "missing_or_invalid_tables": missing_tables,
            "ready": not missing_tables,
        }

    result = {
        "paper": "A Machine-Learned Comorbidity Index",
        "root": str(root.resolve()),
        "required_tables": {
            table: sorted(columns) for table, columns in TABLES.items()
        },
        "candidates": candidates,
        "release_status": release_status,
        "scan_errors": errors,
        "status": (
            "ready"
            if all(item["ready"] for item in release_status.values())
            else "blocked"
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
