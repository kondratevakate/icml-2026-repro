from __future__ import annotations

import math
from pathlib import Path

from audit_glean import (
    accumulation_check,
    clinical_dataset_audit,
    headline_arithmetic,
    proof_audit,
    prose_table_check,
)


ROOT = Path(__file__).resolve().parents[2]
CLINICAL = ROOT / "data" / "mimic-iv-ext-cdm-1.1"


def test_discounted_accumulation_recurrence() -> None:
    result = accumulation_check()
    assert result["absolute_difference"] < 1e-12


def test_official_clinical_payload_and_counts() -> None:
    result = clinical_dataset_audit(CLINICAL)
    assert result["official_checksums"]["all_match"]
    assert result["pathology_total"] == 2400
    assert result["glean_diseases_present"]
    assert result["radiology"]["rows"] == 5960


def test_printed_proof_bound_does_not_follow() -> None:
    result = proof_audit()
    assert not result["printed_term_follows_from_assumption"]
    assert result["supported_numeric_upper_term"] > result["printed_numeric_term"]


def test_headline_and_prose_table_checks() -> None:
    headline = headline_arithmetic()
    assert math.isclose(
        headline["versus_self_consistency"]["auroc_relative_gain"],
        0.12251813101890377,
    )
    assert math.isclose(
        headline["versus_self_consistency"]["brier_relative_reduction"],
        0.5259566620567843,
    )
    assert prose_table_check()["all_values_conflict"]

