#!/usr/bin/env python3
"""Integration tests for both MIMIC cohort paths using synthetic tables."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from build_mimic_cohort import assign_split, build_cohort


def write_fixture(root: Path, release: str) -> dict[str, Path]:
    admissions = pd.DataFrame(
        [
            [1, 101, "2025-01-01", "2025-01-11", None],
            [1, 102, "2025-03-01", "2025-03-03", "2025-03-02"],
            [2, 201, "2025-01-10", "2025-01-18 12:00", None],
            [3, 301, None, None, None],
            [4, 401, "2025-02-01", "2025-02-04", None],
        ],
        columns=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
        ],
    )
    patients = pd.DataFrame(
        [
            [1, "2025-03-02"],
            [2, "2025-03-20"],
            [3, None],
            [4, None],
        ],
        columns=["subject_id", "dod"],
    )
    diagnoses = pd.DataFrame(
        [
            [1, 101, "I50.9", 10],
            [1, 101, " E11 9 ", 10],
            [1, 102, "J18.9", 10],
            [2, 201, "A41-9", 10],
            [3, 301, "N17.9", 10],
            [4, 401, "401.9", 9],
        ],
        columns=["subject_id", "hadm_id", "icd_code", "icd_version"],
    )
    icustays = pd.DataFrame(
        [
            [1, 101, "2025-01-01 12:00"],
            [1, 102, "2025-03-03 12:00"],
            [2, 201, "2025-01-10 00:00"],
            [3, 301, "2025-01-01"],
        ],
        columns=["subject_id", "hadm_id", "intime"],
    )
    paths = {}
    for name, frame in {
        "admissions": admissions,
        "patients": patients,
        "diagnoses": diagnoses,
        "icustays": icustays,
    }.items():
        path = root / f"{name}.csv.gz"
        if name == "diagnoses" and release == "mimic-iii":
            frame = frame.drop(columns=["icd_version"])
        frame.to_csv(path, index=False, compression="gzip")
        paths[name] = path
    return paths


def test_release(root: Path, release: str) -> dict:
    fixture_dir = root / release / "input"
    output_dir = root / release / "output"
    fixture_dir.mkdir(parents=True)
    paths = write_fixture(fixture_dir, release)
    report = build_cohort(
        release=release,
        admissions_path=paths["admissions"],
        patients_path=paths["patients"],
        diagnoses_path=paths["diagnoses"],
        icustays_path=paths["icustays"],
        output_dir=output_dir,
    )
    cohort = pd.read_csv(report["outputs"]["cohort"])
    tokens = pd.read_csv(report["outputs"]["diagnosis_tokens"])

    expected_admissions = 4 if release == "mimic-iv" else 5
    assert len(cohort) == expected_admissions
    assert report["checks"]["no_patient_split_leakage"]
    assert cohort.groupby("subject_id")["split"].nunique().max() == 1
    assert set(tokens["icd_prefix"]) >= {"I509", "E119", "J189", "A419", "N179"}
    if release == "mimic-iv":
        assert 401 not in set(cohort["hadm_id"])
        assert "4019" not in set(tokens["icd_prefix"])
        assert int(cohort.loc[cohort.hadm_id == 101, "icu_transfer"].iloc[0]) == 1
    else:
        assert 401 in set(cohort["hadm_id"])
        assert "4019" in set(tokens["icd_prefix"])
        assert int(cohort.loc[cohort.hadm_id == 101, "icu_transfer"].iloc[0]) == 0
        assert int(cohort.loc[cohort.hadm_id == 102, "icu_transfer"].iloc[0]) == 1

    row_101 = cohort.loc[cohort.hadm_id == 101].iloc[0]
    row_102 = cohort.loc[cohort.hadm_id == 102].iloc[0]
    row_201 = cohort.loc[cohort.hadm_id == 201].iloc[0]
    row_301 = cohort.loc[cohort.hadm_id == 301].iloc[0]
    assert int(row_101.long_stay) == 1
    assert int(row_102.mortality) == 1
    assert int(row_102.mortality_30d) == 1
    assert int(row_201.long_stay) == 1
    assert int(row_201.mortality_30d) == 0
    assert int(row_301.intersection_valid) == 0
    return {
        "release": release,
        "admissions": len(cohort),
        "patients": int(cohort.subject_id.nunique()),
        "tokens": len(tokens),
        "status": report["status"],
    }


def main() -> None:
    assert assign_split(123, algorithm="sha256", salt="") == assign_split(
        123, algorithm="sha256", salt=""
    )
    with tempfile.TemporaryDirectory(prefix="mlci_cohort_fixture_") as directory:
        root = Path(directory)
        results = [
            test_release(root, "mimic-iv"),
            test_release(root, "mimic-iii"),
        ]
    output = {
        "tests": results,
        "checks": {
            "mimic_iv_icd10_filter": True,
            "mimic_iii_late_icu_threshold": True,
            "mimic_iv_any_post_admission_icu_threshold": True,
            "mortality_and_time_labels": True,
            "patient_disjoint_split": True,
            "icd_normalization": True,
        },
        "status": "passed",
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/cohort_fixture.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
