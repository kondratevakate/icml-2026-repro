#!/usr/bin/env python3
"""Build the admission-level MLCI cohort from MIMIC-III or MIMIC-IV."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "admissions": [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "deathtime",
    ],
    "patients": ["subject_id", "dod"],
    "diagnoses": ["subject_id", "hadm_id", "icd_code"],
    "icustays": ["subject_id", "hadm_id", "intime"],
}
PAPER_COUNTS = {
    "mimic-iv": {
        "admissions": 254377,
        "patients": 122905,
        "split_admissions": {"train": 178178, "val": 25228, "test": 50971},
        "split_patients": {"train": 85980, "val": 12235, "test": 24690},
    },
    "mimic-iii": {
        "admissions": 58976,
        "patients": 46520,
        "split_admissions": {"train": 41200, "val": 5867, "test": 11909},
        "split_patients": {"train": 32488, "val": 4630, "test": 9402},
    },
}


def _header(path: Path) -> list[str]:
    if path.name.lower().endswith(".parquet"):
        return [str(column) for column in pd.read_parquet(path).columns]
    return [str(column) for column in pd.read_csv(path, nrows=0).columns]


def read_columns(path: Path, required: list[str]) -> pd.DataFrame:
    available = _header(path)
    by_lower = {column.lower(): column for column in available}
    missing = sorted(set(required) - set(by_lower))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    selected = [by_lower[column] for column in required]
    if path.name.lower().endswith(".parquet"):
        frame = pd.read_parquet(path, columns=selected)
    else:
        frame = pd.read_csv(path, usecols=selected, low_memory=False)
    frame.columns = [column.lower() for column in frame.columns]
    return frame


def normalize_code(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())[:4]


def assign_split(
    subject_id: object,
    *,
    algorithm: str,
    salt: str,
) -> str:
    payload = f"{salt}{subject_id}".encode("utf-8")
    digest = hashlib.new(algorithm, payload).digest()
    unit = int.from_bytes(digest, "big") / (1 << (8 * len(digest)))
    if unit < 0.7:
        return "train"
    if unit < 0.8:
        return "val"
    return "test"


def _datetime(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        frame[column] = pd.to_datetime(
            frame[column], errors="coerce", format="mixed"
        )


def _counts_by_split(frame: pd.DataFrame, unit: str) -> dict[str, int]:
    if unit == "admissions":
        counts = frame["split"].value_counts()
    else:
        counts = (
            frame[["subject_id", "split"]]
            .drop_duplicates()["split"]
            .value_counts()
        )
    return {
        split: int(counts.get(split, 0))
        for split in ("train", "val", "test")
    }


def _label_summary(frame: pd.DataFrame, label: str) -> dict[str, object]:
    valid = frame[f"{label}_defined"].astype(bool)
    values = frame.loc[valid, label]
    return {
        "valid": int(valid.sum()),
        "positive": int(values.sum()),
        "prevalence": float(values.mean()) if len(values) else None,
    }


def build_cohort(
    *,
    release: str,
    admissions_path: Path,
    patients_path: Path,
    diagnoses_path: Path,
    icustays_path: Path,
    output_dir: Path,
    hash_algorithm: str = "sha256",
    hash_salt: str = "",
) -> dict:
    if release not in {"mimic-iii", "mimic-iv"}:
        raise ValueError(f"Unsupported release: {release}")

    diagnosis_columns = list(REQUIRED_COLUMNS["diagnoses"])
    if release == "mimic-iv":
        diagnosis_columns.append("icd_version")
    admissions = read_columns(admissions_path, REQUIRED_COLUMNS["admissions"])
    patients = read_columns(patients_path, REQUIRED_COLUMNS["patients"])
    diagnoses = read_columns(diagnoses_path, diagnosis_columns)
    icustays = read_columns(icustays_path, REQUIRED_COLUMNS["icustays"])

    for frame in (admissions, patients, diagnoses, icustays):
        frame["subject_id"] = pd.to_numeric(
            frame["subject_id"], errors="raise"
        ).astype("int64")
    for frame in (admissions, diagnoses, icustays):
        frame["hadm_id"] = pd.to_numeric(
            frame["hadm_id"], errors="raise"
        ).astype("int64")

    if release == "mimic-iv":
        version = pd.to_numeric(diagnoses["icd_version"], errors="coerce")
        diagnoses = diagnoses.loc[version.eq(10)].copy()
    diagnoses["icd_prefix"] = diagnoses["icd_code"].map(normalize_code)
    diagnoses = diagnoses.loc[diagnoses["icd_prefix"].ne("")].copy()

    cohort_keys = diagnoses[["subject_id", "hadm_id"]].drop_duplicates()
    cohort = admissions.merge(
        cohort_keys, on=["subject_id", "hadm_id"], how="inner", validate="one_to_one"
    )
    cohort = cohort.merge(
        patients[["subject_id", "dod"]],
        on="subject_id",
        how="left",
        validate="many_to_one",
    )

    _datetime(cohort, ["admittime", "dischtime", "deathtime", "dod"])
    _datetime(icustays, ["intime"])
    first_icu = (
        icustays.dropna(subset=["intime"])
        .groupby(["subject_id", "hadm_id"], as_index=False)["intime"]
        .min()
        .rename(columns={"intime": "first_icu_intime"})
    )
    cohort = cohort.merge(
        first_icu,
        on=["subject_id", "hadm_id"],
        how="left",
        validate="one_to_one",
    )

    cohort["mortality"] = cohort["deathtime"].notna().astype("int8")
    cohort["mortality_defined"] = True

    cohort["mortality_30d_defined"] = cohort["admittime"].notna()
    death_days = (cohort["dod"] - cohort["admittime"]).dt.total_seconds() / 86400
    cohort["mortality_30d"] = (
        death_days.between(0, 30, inclusive="both")
        & cohort["mortality_30d_defined"]
    ).astype("int8")

    cohort["long_stay_defined"] = (
        cohort["admittime"].notna() & cohort["dischtime"].notna()
    )
    stay_days = (
        cohort["dischtime"] - cohort["admittime"]
    ).dt.total_seconds() / 86400
    cohort["long_stay"] = (
        stay_days.gt(7) & cohort["long_stay_defined"]
    ).astype("int8")

    cohort["icu_transfer_defined"] = cohort["admittime"].notna()
    time_to_icu = (
        cohort["first_icu_intime"] - cohort["admittime"]
    ).dt.total_seconds() / 3600
    threshold_hours = 0 if release == "mimic-iv" else 24
    cohort["icu_transfer"] = (
        time_to_icu.gt(threshold_hours) & cohort["icu_transfer_defined"]
    ).astype("int8")

    cohort["intersection_valid"] = (
        cohort[
            [
                "mortality_defined",
                "mortality_30d_defined",
                "long_stay_defined",
                "icu_transfer_defined",
            ]
        ]
        .all(axis=1)
        .astype("int8")
    )
    cohort["split"] = cohort["subject_id"].map(
        lambda value: assign_split(
            value, algorithm=hash_algorithm, salt=hash_salt
        )
    )

    token_rows = diagnoses.merge(
        cohort[["subject_id", "hadm_id", "split"]],
        on=["subject_id", "hadm_id"],
        how="inner",
        validate="many_to_one",
    )[["subject_id", "hadm_id", "split", "icd_prefix"]]

    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = output_dir / f"{release}_cohort.csv.gz"
    token_path = output_dir / f"{release}_diagnosis_tokens.csv.gz"
    cohort.sort_values(["subject_id", "hadm_id"]).to_csv(
        cohort_path, index=False, compression="gzip"
    )
    token_rows.to_csv(token_path, index=False, compression="gzip")

    patient_splits = cohort[["subject_id", "split"]].drop_duplicates()
    leakage = int(
        patient_splits.groupby("subject_id")["split"].nunique().gt(1).sum()
    )
    split_admissions = _counts_by_split(cohort, "admissions")
    split_patients = _counts_by_split(cohort, "patients")
    paper = PAPER_COUNTS[release]
    report = {
        "release": release,
        "inputs": {
            "admissions": str(admissions_path.resolve()),
            "patients": str(patients_path.resolve()),
            "diagnoses": str(diagnoses_path.resolve()),
            "icustays": str(icustays_path.resolve()),
        },
        "outputs": {
            "cohort": str(cohort_path.resolve()),
            "diagnosis_tokens": str(token_path.resolve()),
        },
        "split_protocol": {
            "algorithm": hash_algorithm,
            "salt": hash_salt,
            "proportions": {"train": 0.7, "val": 0.1, "test": 0.2},
            "paper_algorithm_and_salt_disclosed": False,
        },
        "counts": {
            "admissions": int(len(cohort)),
            "patients": int(cohort["subject_id"].nunique()),
            "diagnosis_tokens": int(len(token_rows)),
            "intersection_valid": int(cohort["intersection_valid"].sum()),
            "split_admissions": split_admissions,
            "split_patients": split_patients,
        },
        "paper_counts": paper,
        "paper_count_differences": {
            "admissions": int(len(cohort) - paper["admissions"]),
            "patients": int(
                cohort["subject_id"].nunique() - paper["patients"]
            ),
            "split_admissions": {
                split: split_admissions[split]
                - paper["split_admissions"][split]
                for split in split_admissions
            },
            "split_patients": {
                split: split_patients[split]
                - paper["split_patients"][split]
                for split in split_patients
            },
        },
        "labels": {
            label: _label_summary(cohort, label)
            for label in (
                "mortality",
                "mortality_30d",
                "long_stay",
                "icu_transfer",
            )
        },
        "checks": {
            "unique_admissions": not cohort.duplicated(
                ["subject_id", "hadm_id"]
            ).any(),
            "no_patient_split_leakage": leakage == 0,
            "all_admissions_have_tokens": int(
                cohort_keys.merge(
                    cohort[["subject_id", "hadm_id"]],
                    on=["subject_id", "hadm_id"],
                    how="right",
                    indicator=True,
                )["_merge"].eq("both").sum()
            )
            == len(cohort),
            "all_prefixes_nonempty_and_at_most_four": bool(
                token_rows["icd_prefix"].str.len().between(1, 4).all()
            ),
        },
    }
    report["status"] = (
        "passed"
        if all(bool(value) for value in report["checks"].values())
        else "failed"
    )
    report_path = output_dir / f"{release}_cohort_report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", choices=["mimic-iii", "mimic-iv"], required=True)
    parser.add_argument("--admissions", type=Path, required=True)
    parser.add_argument("--patients", type=Path, required=True)
    parser.add_argument("--diagnoses", type=Path, required=True)
    parser.add_argument("--icustays", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--hash-algorithm",
        choices=sorted(hashlib.algorithms_guaranteed),
        default="sha256",
    )
    parser.add_argument("--hash-salt", default="")
    args = parser.parse_args()
    report = build_cohort(
        release=args.release,
        admissions_path=args.admissions,
        patients_path=args.patients,
        diagnoses_path=args.diagnoses,
        icustays_path=args.icustays,
        output_dir=args.output_dir,
        hash_algorithm=args.hash_algorithm,
        hash_salt=args.hash_salt,
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
