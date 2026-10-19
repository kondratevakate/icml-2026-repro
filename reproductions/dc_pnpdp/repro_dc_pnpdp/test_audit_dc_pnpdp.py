from audit_dc_pnpdp import (
    cg_penalty_check,
    release_inventory,
    scalar_fixed_point_check,
    sh_check,
)


def test_dual_fixed_point_satisfies_original_objective():
    result = scalar_fixed_point_check()["dual"]
    assert result["consensus_residual"] < 1e-12
    assert result["data_update_residual"] < 1e-12
    assert result["prox_update_residual"] < 1e-12
    assert result["original_objective_residual"] < 1e-12


def test_loose_fixed_point_is_moreau_not_original_objective():
    result = scalar_fixed_point_check()["loose"]
    assert result["fixed_point_equation_residual"] < 1e-12
    assert result["moreau_envelope_objective_residual"] < 1e-12
    assert result["original_objective_residual"] > 1e-2


def test_zero_release_penalty_removes_anchor_from_exact_identity_solve():
    result = cg_penalty_check()
    assert result["zero_penalty_anchor_difference"] < 1e-12
    assert result["zero_penalty_solution_error_vs_y"] < 1e-12
    assert result["positive_penalty_anchor_difference"] > 0.1


def test_spectral_homogenization_smoke():
    result = sh_check(trials=16)
    assert result["shape_preserved"]
    assert result["dtype_preserved"]
    assert result["finite"]


def test_released_smoothing_breaks_fft_hermitian_symmetry():
    result = sh_check(trials=16)
    assert result["complement_hermitian_asymmetry_ratio"] > 1e-4


def test_release_inventory_and_compile():
    result = release_inventory()
    assert result["python_compile_errors"] == {}
    assert result["has_mri_implementation"] is False
    assert result["has_cached_outputs"] is False
    assert result["has_tests"] is False
    assert result["has_requirements"] is False

