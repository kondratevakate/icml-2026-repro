from __future__ import annotations

import math

from audit_caml import (
    aligned_constraint_check,
    artifact_audit,
    degenerate_offset_check,
    release_conformance,
    theorem_counterexample,
)


def test_artifact_identity() -> None:
    data = artifact_audit()
    assert data["paper"]["matches"]
    assert data["source"]["matches"]
    assert data["repository"]["commit_matches"]
    assert data["repository"]["result_files"] == []


def test_nonunique_does_not_imply_connected_valley() -> None:
    data = theorem_counterexample()
    assert data["non_unique"]
    assert data["solutions_are_isolated"]
    assert not data["solution_set_connected"]
    assert not data["headline_implication_valid"]


def test_linear_offset_matches_closed_form() -> None:
    data = aligned_constraint_check()
    assert data["absolute_difference"] < 1e-12
    assert data["offset_detached"]
    assert data["total_is_finite"]


def test_delay_ramp_schedule() -> None:
    data = aligned_constraint_check()
    for key, expected in data["expected_schedule"].items():
        assert math.isclose(data["schedule"][key], expected, abs_tol=1e-12)


def test_degenerate_linear_offset_is_unhandled() -> None:
    data = degenerate_offset_check()
    assert not data["is_finite"]
    assert not data["denominator_guard_present"]


def test_release_config_does_not_match_paper() -> None:
    data = release_conformance()
    observed = {
        (item["benchmark"], item["parameter"])
        for item in data["mismatches"]
    }
    assert observed == {
        ("heat", "w_bc"),
        ("heat", "target_l2"),
        ("ns", "target_l2"),
        ("helm", "w_bc"),
        ("helm", "min_epochs"),
    }
    assert data["readme_claims_only_standard_torch"]
    assert data["undocumented_overrides_import"]
