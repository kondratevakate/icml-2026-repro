from __future__ import annotations

from audit_lvcg import (
    artifact_audit,
    conformance_audit,
    geometry_check,
    runtime_import_audit,
    split_audit,
    table_arithmetic,
)


def test_official_artifact_identity() -> None:
    data = artifact_audit()
    assert data["paper"]["matches"]
    assert data["source"]["matches"]
    assert data["repository"]["commit_matches"]
    assert data["repository"]["checkpoints"] == []
    assert data["repository"]["result_files"] == []


def test_geometry_equations_independently() -> None:
    data = geometry_check()
    assert data["passes_1e_4"]


def test_release_import_failure_is_reproduced() -> None:
    data = runtime_import_audit()
    assert not data["succeeded"]
    assert data["last_error_line"] == "ModuleNotFoundError: No module named 'lvcg.data'"
    assert {item["resolved"] for item in data["missing_relative_imports"]} == {
        "lvcg/data/angle",
        "lvcg/data/beat_segmentation",
    }


def test_released_splits_are_disjoint() -> None:
    data = split_audit()
    assert data["all_record_splits_disjoint"]
    assert data["all_available_patient_splits_disjoint"]
    assert all(
        all(count == 0 for count in task["duplicate_records"].values())
        for task in data["tasks"].values()
    )
    assert not data["tasks"]["ptbxl_sub_class"]["column_schema_equal"]


def test_table_average_arithmetic() -> None:
    assert table_arithmetic()["agrees_at_printed_precision"]


def test_paper_release_conformance_gaps() -> None:
    data = conformance_audit()
    assert data["paper_objective_has_three_terms"]
    assert data["released_training_adds_base_loss"]
    assert data["checkpoint_override_is_not_serialized"]
    assert data["paper_noncardiac_dataset"] != data["released_noncardiac_dataset"]
    assert data["paper_source_mentions_mimic_ext_icd"]
    assert data["release_config_mentions_aireadi"]
    assert not data["release_config_mentions_mimic_ext_icd"]
