from audit_bayes_causal_meta import (
    paper_lipschitz_check,
    release_inventory,
    released_prior_check,
    theorem_constant_check,
)


def test_release_prior_is_scale_invariant_for_positive_scalar_z():
    result = released_prior_check()
    assert result["scale_invariance_error"] == 0.0


def test_release_prior_is_discontinuous_at_zero():
    result = released_prior_check()
    assert result["jump_norm_at_zero"] == 1.0
    assert result["released_map_violates_linear_bound"]


def test_paper_gaussian_lipschitz_bound():
    result = paper_lipschitz_check()
    assert result["all_bounds_hold"]
    assert result["error_decomposition_is_triangle_inequality"]


def test_appendix_constant_drops_condition_number():
    result = theorem_constant_check()
    assert result["appendix_embedding_condition_holds"]
    assert not result["required_parameter_condition_holds"]
    assert result["counterexample"]


def test_release_inventory_and_compile():
    result = release_inventory()
    assert result["python_compile_errors"] == {}
    assert result["has_tests"] is False
    assert result["has_cached_metrics_or_checkpoints"] is False
    assert result["has_ukbb_data"] is False


def test_release_does_not_seed_training_rngs():
    result = release_inventory()
    assert result["main_sets_torch_seed"] is False
    assert result["main_sets_numpy_seed"] is False

