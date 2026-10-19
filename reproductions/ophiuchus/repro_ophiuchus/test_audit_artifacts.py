from audit_artifacts import audit_code_spec, audit_release_data, audit_state_machine


def test_release_contains_examples_not_64k_corpus():
    evidence = audit_release_data()
    assert evidence["sft_records"] == 2
    assert evidence["rl_records"] == 2
    assert evidence["train_test_parquet_identical"]
    assert not evidence["complete_64k_corpus_present"]


def test_tool_loop_smoke_and_release_format_mismatch():
    evidence = audit_state_machine()
    assert evidence["canonical_action_tool_call_parsed"]
    assert evidence["final_won"]
    assert evidence["final_reward"] == 4.0
    assert not evidence["sft_style_tool_call_parsed"]


def test_duplicate_early_stop_and_mask_schema_are_not_implemented():
    evidence = audit_state_machine()
    assert not evidence["duplicate_zoom_early_stopped"]
    assert evidence["released_list_mask_reset_error"].startswith("TypeError:")


def test_three_stages_exist_but_canonical_run_is_not_paper_faithful():
    checks = audit_code_spec()["checks"]
    assert checks["cold_start_script_present"]
    assert checks["reflection_script_present"]
    assert checks["atrl_script_present"]
    assert checks["preprocess_contains_invalid_placeholder_expression"]
    assert checks["paper_grpo_without_kl"]
    assert checks["atrl_effective_kl_loss_enabled"]
