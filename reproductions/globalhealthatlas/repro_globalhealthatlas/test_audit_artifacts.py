from audit_artifacts import build_audit


def test_adapter_payloads_match_hub_and_are_well_formed():
    audit = build_audit()
    for adapter in audit["adapters"].values():
        assert adapter["hub_checksum_matches"]
        assert adapter["payload_extent_matches_file"]
        assert adapter["tensor_count"] > 0
        assert adapter["parameter_count"] > 0


def test_detailed_benchmark_reconstruction():
    result = build_audit()["benchmark_reconstruction"]
    assert result["models_compared"] == 16
    assert result["rows_matching_paper_rounding"] == 16
    assert result["max_score_abs_error"] < 5e-4


def test_cached_experiment_arithmetic():
    audit = build_audit()
    assert audit["incremental_reconstruction"]["max_abs_error"] < 1e-5
    assert audit["transfer_reconstruction"]["rows_matching_paper_rounding"] == 11
    assert audit["transfer_reconstruction"]["max_abs_error_after_paper_rounding"] > 0.13
    assert audit["robustness_reconstruction"]["max_abs_error"] == 0
    assert audit["leakage_reconstruction"]["all_10gram_zero"]


def test_lightweight_released_code_smoke():
    smoke = build_audit()["lightweight_code_smoke"]
    assert smoke["required_values_present"]
    assert not smoke["placeholders_remaining"]
    assert smoke["atomic_json_roundtrip"]
    assert not smoke["stale_temp_exists"]
