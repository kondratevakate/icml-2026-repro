from audit_artifacts import (
    audit_checkpoint,
    audit_code_and_spec,
    audit_synthetic_data,
)


def test_synthetic_h5_and_metadata_align():
    evidence = audit_synthetic_data()
    assert evidence["passed"]
    assert evidence["ppg_shape"] == [3000, 1000]
    assert evidence["subjects"] == 60
    assert evidence["label_noise_rows"] == 150


def test_checkpoint_strict_loads_but_eval_is_nonfinite():
    evidence = audit_checkpoint()
    assert evidence["strict_load_passed"]
    assert evidence["nonfinite_checkpoint_values"] == 0
    assert evidence["affected_negative_running_var_tensors"] > 0
    assert evidence["train_mode_embedding_finite"]
    assert not evidence["eval_mode_embedding_finite"]


def test_core_architecture_is_present_but_curriculum_does_not_match():
    evidence = audit_code_and_spec()
    assert evidence["method_core_present"]
    assert not evidence["method_matches_paper"]
    assert evidence["canonical_mask_ratio_ecg"] == 0.8
    assert not evidence["canonical_use_cl"]


def test_notebook_paths_are_not_checkpoint_backed_evaluation():
    checks = audit_code_and_spec()["checks"]
    assert checks["simple_notebook_loads_checkpoint"]
    assert not checks["simple_notebook_uses_true_ecg_tensor"]
    assert not checks["pvc_notebook_loads_checkpoint"]
