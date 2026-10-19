import json
from pathlib import Path


EVIDENCE = Path(__file__).resolve().parent / "evidence" / "artifact_audit.json"


def load():
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_ratio_ode_oracle():
    data = load()
    assert data["runtime"]["oracle_max_abs_error"] < 1e-5


def test_small_gaussian_run():
    run = load()["runtime"]["small_training"]
    assert run["finite"]
    assert run["direct_mse"] < run["naive_mse"]
    assert run["direct_seconds"] > 0
    assert run["naive_seconds"] > 0


def test_cached_gaussian_provenance():
    result = load()["gaussian"]
    assert result["all_direct_faster_than_naive"]
    assert result["all_direct_mse_lower_than_naive"]
    assert result["scratio_cached_values_per_row"] == [3]
    assert result["paper_caption_training_runs"] == 5
    assert not result["caption_vs_cached_run_count_match"]


def test_application_evidence_boundaries():
    data = load()
    assert data["mutual_information"]["rounded_values_matching"] < 5
    assert data["differential_abundance"]["scratio_leads_metrics"] == 5
    assert data["batch_correction"]["all_executed_shapes_present"]
    assert data["treatment_response"]["combosciplex_pearson"] > 0.9
    assert data["prepared_score"] == 7
