#!/usr/bin/env python3
"""Independent artifact audit for the official scRatio release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
DEFAULT_OFFICIAL = CANDIDATE / "official"
EXPECTED_ZENODO_BYTES = 3_302_638_232
EXPECTED_ZENODO_MD5 = "4fc167ca536390b39edb037398c84914"
SCHEDULES = {(0.0, 0.0), (0.0, 0.001), (0.1, 0.0)}


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_array(value: str) -> np.ndarray:
    return np.fromstring(value.strip().strip("[]"), sep=" ")


def notebook(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def output_text(cell: dict[str, Any]) -> str:
    blocks: list[str] = []
    for output in cell.get("outputs", []):
        if output.get("text"):
            blocks.append("".join(output["text"]))
        plain = output.get("data", {}).get("text/plain")
        if plain:
            blocks.append("".join(plain))
    return "\n".join(blocks)


def find_output(nb: dict[str, Any], source_fragment: str) -> str:
    for cell in nb.get("cells", []):
        if source_fragment in "".join(cell.get("source", [])):
            return output_text(cell)
    raise KeyError(source_fragment)


def parse_metric_output(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        match = re.match(r"\s*([A-Za-z_]+)\s+([-+]?\d+(?:\.\d+)?)", line)
        if match:
            metrics[match.group(1)] = float(match.group(2))
    return metrics


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def repository_inventory(repo: Path) -> dict[str, Any]:
    tracked = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files"], text=True, encoding="utf-8"
    ).splitlines()
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()
    commit_date = subprocess.check_output(
        ["git", "-C", str(repo), "show", "-s", "--format=%cI", "HEAD"],
        text=True,
        encoding="utf-8",
    ).strip()
    suffixes = Counter(Path(name).suffix.lower() or "<none>" for name in tracked)
    result_like = [
        name
        for name in tracked
        if Path(name).suffix.lower() in {".ckpt", ".pt", ".pth", ".npy", ".npz", ".pkl", ".h5ad"}
    ]
    return {
        "commit": commit,
        "commit_date": commit_date,
        "tracked_files": len(tracked),
        "tracked_bytes": sum((repo / name).stat().st_size for name in tracked),
        "suffix_counts": dict(sorted(suffixes.items())),
        "tracked_checkpoints_or_raw_predictions": result_like,
        "tracked_notebooks": suffixes[".ipynb"],
        "tracked_csvs": suffixes[".csv"],
        "has_tests_directory": (repo / "tests").is_dir(),
    }


def gaussian_audit(repo: Path, paper_source: Path) -> dict[str, Any]:
    path = repo / "notebooks" / "gaussian_tests" / "all_results.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    selected = [
        row
        for row in rows
        if float(row["loc"]) in {1.0, 2.0}
        and (float(row["sigma"]), float(row["sigma_min"])) in SCHEDULES
    ]
    grouped: dict[tuple[float, int, float, float], dict[str, dict[str, str]]] = {}
    for row in selected:
        key = (
            float(row["loc"]),
            int(row["num_dims"]),
            float(row["sigma"]),
            float(row["sigma_min"]),
        )
        grouped.setdefault(key, {})[row["score_type"]] = row

    comparisons = []
    for key, group in sorted(grouped.items()):
        if "naive" not in group or "rl" not in group:
            continue
        naive, ratio = group["naive"], group["rl"]
        naive_time = float(parse_array(naive["time"]).mean())
        ratio_time = float(parse_array(ratio["time"]).mean())
        naive_mse = float(naive["mean_value"])
        ratio_mse = float(ratio["mean_value"])
        comparisons.append(
            {
                "loc": key[0],
                "dimension": key[1],
                "sigma": key[2],
                "sigma_min": key[3],
                "speedup": naive_time / ratio_time,
                "mse_gain": naive_mse / ratio_mse,
            }
        )

    sc_rows = [row for row in rows if row["score_type"] in {"naive", "rl"}]
    values_per_row = sorted({len(parse_array(row["value"])) for row in sc_rows})
    time_values_per_row = sorted({len(parse_array(row["time"])) for row in sc_rows})
    caption = (paper_source / "figure_tex" / "mse_time_plot.tex").read_text(encoding="utf-8")
    caption_runs_match = re.search(r"average MSE across\s+(\d+)\s+training runs", caption)
    caption_runs = int(caption_runs_match.group(1)) if caption_runs_match else None

    baseline_path = repo / "notebooks" / "gaussian_tests" / "results_time_score_matching_sweep.csv"
    with baseline_path.open(newline="", encoding="utf-8") as handle:
        baseline_rows = list(csv.DictReader(handle))
    baseline_run_counts = Counter()
    for row in baseline_rows:
        key = (
            row["Dimension"],
            row["Location"],
            row["Score Type"],
            row["Learning Rate"],
            row["SB Variance"],
        )
        baseline_run_counts[key] += 1

    speedups = np.asarray([entry["speedup"] for entry in comparisons])
    mse_gains = np.asarray([entry["mse_gain"] for entry in comparisons])
    return {
        "cached_rows": len(rows),
        "selected_schedule_comparisons": len(comparisons),
        "all_direct_faster_than_naive": bool(np.all(speedups > 1)),
        "speedup_min_median_max": [
            float(speedups.min()),
            float(np.median(speedups)),
            float(speedups.max()),
        ],
        "all_direct_mse_lower_than_naive": bool(np.all(mse_gains > 1)),
        "mse_gain_min_median_max": [
            float(mse_gains.min()),
            float(np.median(mse_gains)),
            float(mse_gains.max()),
        ],
        "scratio_cached_values_per_row": values_per_row,
        "scratio_cached_times_per_row": time_values_per_row,
        "paper_caption_training_runs": caption_runs,
        "caption_vs_cached_run_count_match": caption_runs in values_per_row,
        "baseline_runs_per_configuration": sorted(set(baseline_run_counts.values())),
    }


def notebook_audit(repo: Path) -> dict[str, Any]:
    notebooks = sorted((repo / "notebooks").rglob("*.ipynb"))
    target_tokens = (
        "mi_estimation",
        "differential_abundance",
        "neurips",
        "celegans",
        "combosciplex",
        "pbmc10m",
    )
    summaries = []
    for path in notebooks:
        if not any(token in str(path).lower() for token in target_tokens):
            continue
        nb = notebook(path)
        code_cells = [cell for cell in nb.get("cells", []) if cell.get("cell_type") == "code"]
        errors = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        summaries.append(
            {
                "path": path.relative_to(repo).as_posix(),
                "code_cells": len(code_cells),
                "cells_with_outputs": sum(bool(cell.get("outputs")) for cell in code_cells),
                "error_outputs": len(errors),
            }
        )
    return {
        "target_notebooks": len(summaries),
        "notebooks_with_error_outputs": sum(item["error_outputs"] > 0 for item in summaries),
        "notebooks": summaries,
    }


def mi_audit(repo: Path) -> dict[str, Any]:
    nb = notebook(repo / "notebooks" / "mi_estimation" / "evaluate_results.ipynb")
    mean_output = find_output(nb, 'results_df.groupby(["Config", "Dimensions"]).mean()')
    block = mean_output.split("stochastic_0.25", 1)[1].split("stochastic_0.5", 1)[0]
    values = {}
    for match in re.finditer(r"^\s*(20|40|80|160|320)\s+([-+]?\d+\.\d+)", block, re.MULTILINE):
        values[int(match.group(1))] = float(match.group(2))
    if len(values) != 5:
        raise RuntimeError(f"Could not parse MI stochastic_0.25 output: {values}")
    paper_values = {20: 0.03, 40: 0.09, 80: 0.07, 160: 0.11, 320: 1.16}
    return {
        "notebook_stochastic_0.25_mae": values,
        "paper_table_stochastic_0.25_mae": paper_values,
        "rounded_values_matching": sum(
            math.isclose(values[dimension], paper_values[dimension], abs_tol=0.005)
            for dimension in paper_values
        ),
        "largest_absolute_difference": max(
            abs(values[dimension] - paper_values[dimension]) for dimension in paper_values
        ),
        "raw_ratio_arrays_tracked": False,
        "note": "The executed notebook points to external project_folder/results/*.npy paths.",
    }


def differential_abundance_audit(repo: Path) -> dict[str, Any]:
    nb = notebook(
        repo / "notebooks" / "differential_abundance_analysis" / "evaluate_models.ipynb"
    )
    sources = {
        "MrVI": 'results_mrvi_per_run.groupby("metric").mean()',
        "MELD": 'results_meld_per_run.groupby("metric").mean()',
        "scRatio deterministic": 'results_scratio_det_per_run.groupby("metric").mean()',
        "scRatio sigma_min=0.1": 'results_scratio_det_sigma_min_per_run.groupby("metric").mean()',
        "scRatio stochastic": 'results_scratio_stochastic_per_run.groupby("metric").mean()',
    }
    keys = (
        "corr_auc_score_p",
        "mean_auc_score",
        "corr_non_abundant_over_abundant_p",
        "mean_non_abundant_over_abundant",
        "corr_correct_sign_prop_p",
        "mean_correct_sign_prop",
    )
    metrics = {
        name: {key: parse_metric_output(find_output(nb, fragment))[key] for key in keys}
        for name, fragment in sources.items()
    }
    rounded = {
        name: {key: round(value, 2) for key, value in model_metrics.items()}
        for name, model_metrics in metrics.items()
    }
    leaders = {
        key: max(metrics, key=lambda name: metrics[name][key])
        for key in keys
    }
    return {
        "executed_notebook_rounded_metrics": rounded,
        "metric_leaders": leaders,
        "scratio_leads_metrics": sum(name.startswith("scRatio") for name in leaders.values()),
        "metrics_checked": len(keys),
        "raw_metric_inputs_tracked": False,
    }


def batch_audit(repo: Path) -> dict[str, Any]:
    details = {}
    for dataset, rel, expected_rows, expected_groups in (
        ("NeurIPS", "Neurips/generate_concatenated_plots.ipynb", 90_261, 431),
        ("C_elegans", "cElegans/generate_concatenated_plots.ipynb", 89_701, 248),
    ):
        nb = notebook(repo / "notebooks" / rel)
        all_text = "\n".join(output_text(cell) for cell in nb.get("cells", []))
        shape_match = f"llr_df.shape=({expected_rows}, 12)" in all_text
        group_count_present = f"/{expected_groups}" in all_text
        details[dataset] = {
            "expected_cells": expected_rows,
            "executed_shape_present": shape_match,
            "expected_condition_groups": expected_groups,
            "condition_group_progress_present": group_count_present,
        }
    return {
        "datasets": details,
        "all_executed_shapes_present": all(
            item["executed_shape_present"] for item in details.values()
        ),
        "raw_llr_npz_files_tracked": False,
        "numeric_decrease_recomputable_from_clone": False,
    }


def treatment_audit(repo: Path) -> dict[str, Any]:
    combo_nb = notebook(repo / "notebooks" / "combosciplex" / "scRatio_ae_5.ipynb")
    text = "\n".join(output_text(cell) for cell in combo_nb.get("cells", []))
    log_odds = np.asarray(
        [float(value) for value in re.findall(r"Log-odds:\s*([-+0-9.eE]+)", text)]
    )
    log_ratios = np.asarray(
        [float(value) for value in re.findall(r"Log log-ratio:\s*([-+0-9.eE]+)", text)]
    )
    if len(log_odds) != len(log_ratios) or len(log_odds) < 2:
        raise RuntimeError("Could not recover paired ComboSciPlex outputs")
    pearson = float(np.corrcoef(log_ratios, log_odds)[0, 1])
    spearman = float(np.corrcoef(rankdata(log_ratios), rankdata(log_odds))[0, 1])

    pbmc_nb = notebook(repo / "notebooks" / "pbmc10m" / "scRatio_pbmc10m.ipynb")
    pbmc_source = "\n".join("".join(cell.get("source", [])) for cell in pbmc_nb.get("cells", []))
    pbmc_text = "\n".join(output_text(cell) for cell in pbmc_nb.get("cells", []))
    return {
        "combosciplex_pairs_recovered": len(log_odds),
        "combosciplex_pearson": pearson,
        "combosciplex_spearman": spearman,
        "pbmc_expected_labels_present": all(
            label in pbmc_source for label in ("IFN-omega", "APRIL", "IL-10")
        ),
        "pbmc_executed_figure_outputs": pbmc_text.count("<Figure size"),
        "raw_combo_or_pbmc_predictions_tracked": False,
    }


def zenodo_audit(official: Path) -> dict[str, Any]:
    metadata = json.loads((official / "zenodo-20822377.json").read_text(encoding="utf-8-sig"))
    record_file = metadata["files"][0]
    archive = official / record_file["key"]
    size = archive.stat().st_size if archive.exists() else 0
    complete = size == EXPECTED_ZENODO_BYTES
    result: dict[str, Any] = {
        "record_id": metadata["id"],
        "title": metadata["metadata"]["title"],
        "file": record_file["key"],
        "expected_bytes": EXPECTED_ZENODO_BYTES,
        "local_bytes": size,
        "complete": complete,
        "declared_md5": EXPECTED_ZENODO_MD5,
    }
    if complete:
        result["local_md5"] = file_hash(archive, "md5")
        result["checksum_match"] = result["local_md5"] == EXPECTED_ZENODO_MD5
        with zipfile.ZipFile(archive) as handle:
            infos = handle.infolist()
        result["zip_members"] = len(infos)
        result["zip_uncompressed_bytes"] = sum(info.file_size for info in infos)
        result["top_level_entries"] = sorted(
            {info.filename.replace("\\", "/").split("/", 1)[0] for info in infos}
        )
    return result


def runtime_checks(repo: Path, include_small_training: bool) -> dict[str, Any]:
    import lightning as light
    import torch
    from scRatio.models.flow_matching import ConditionalFlowMatchingWithScore

    class Oracle(ConditionalFlowMatchingWithScore):
        def __init__(self, sigma_min: float = 0.1):
            light.LightningModule.__init__(self)
            self.sigma_min = sigma_min

        def forward(self, x, t, cond, use_conds=None):
            if not torch.is_tensor(t):
                t = torch.tensor(t, dtype=x.dtype, device=x.device)
            if t.ndim == 0:
                t = t.expand(x.shape[0]).unsqueeze(1)
            elif t.ndim == 1:
                t = t.unsqueeze(1)
            mean_vector = cond[:, :1].expand_as(x)
            lam = 1 - (1 - self.sigma_min) * t
            lam_dot = -(1 - self.sigma_min)
            variance = t * t + lam * lam
            mean = t * mean_vector
            score = -(x - mean) / variance
            variance_dot = 2 * t + 2 * lam * lam_dot
            field = mean_vector + 0.5 * variance_dot / variance * (x - mean)
            return field, score

    points = torch.tensor([[0.2, -0.3], [1.2, 0.8], [-1.0, 0.5]], dtype=torch.float64)
    condition = torch.ones((3, 1), dtype=torch.float64)
    control = torch.zeros((3, 1), dtype=torch.float64)
    oracle = Oracle(0.1)
    predicted = oracle.estimate_log_density_ratio(
        points,
        condition,
        control,
        condition,
        n_steps=101,
        estimator_type="exact",
        solver="dopri5",
    )
    true = ((points.sum(1) - points.shape[1] / 2) / 1.01).numpy()
    result: dict[str, Any] = {
        "torch_version": torch.__version__,
        "oracle_predicted": predicted.tolist(),
        "oracle_true": true.tolist(),
        "oracle_max_abs_error": float(np.max(np.abs(predicted - true))),
    }
    if not include_small_training:
        return result

    torch.manual_seed(260224201)
    np.random.seed(260224201)
    sigma_min = 0.1
    lambda_t = lambda t: 1 - (1 - sigma_min) * t
    lambda_sp_t = lambda t: torch.full_like(t, -(1 - sigma_min))
    model = ConditionalFlowMatchingWithScore(
        input_dim=2,
        cond_dims=[2],
        hidden_dims=[64, 64],
        encoder_hidden_dims=[32],
        encoder_out_dim=16,
        encoder_out_dim_cond=8,
        time_feature_dim=16,
        lambda_t=lambda_t,
        lambda_sp_t=lambda_sp_t,
        betas=[0.0],
        lr=1e-3,
    )
    optimizer = model.configure_optimizers()
    start = time.perf_counter()
    for _ in range(1200):
        labels = torch.randint(0, 2, (256,))
        samples = torch.randn(256, 2) + labels[:, None]
        conditions = torch.nn.functional.one_hot(labels, 2).float()
        optimizer.zero_grad()
        loss = model.shared_step(samples, conditions)
        loss.backward()
        optimizer.step()
    train_seconds = time.perf_counter() - start
    samples = torch.randn(128, 2) + 1
    condition = torch.tensor([[0.0, 1.0]]).expand(128, -1)
    control = torch.tensor([[1.0, 0.0]]).expand(128, -1)
    start = time.perf_counter()
    direct = model.estimate_log_density_ratio(
        samples, condition, control, condition, n_steps=31, estimator_type="exact", solver="rk4"
    )
    direct_seconds = time.perf_counter() - start
    start = time.perf_counter()
    naive = model.estimate_log_density(
        samples, condition, n_steps=31, estimator_type="exact", solver="rk4"
    ) - model.estimate_log_density(
        samples, control, n_steps=31, estimator_type="exact", solver="rk4"
    )
    naive_seconds = time.perf_counter() - start
    true = (samples.sum(1) - 1).numpy()
    result["small_training"] = {
        "seed": 260224201,
        "steps": 1200,
        "train_seconds": train_seconds,
        "direct_mse": float(np.mean((direct - true) ** 2)),
        "naive_mse": float(np.mean((naive - true) ** 2)),
        "direct_seconds": direct_seconds,
        "naive_seconds": naive_seconds,
        "speedup": naive_seconds / direct_seconds,
        "finite": bool(np.isfinite(direct).all() and np.isfinite(naive).all()),
    }
    return result


def build_audit(official: Path, run_runtime: bool, small_training: bool) -> dict[str, Any]:
    repo = official / "scRatio"
    source = official / "paper_source"
    audit = {
        "paper": {
            "openreview_id": "5zbPdMNcl9",
            "arxiv_id": "2602.24201v2",
            "pdf_sha256": file_hash(official / "2602.24201.pdf"),
            "source_sha256": file_hash(official / "2602.24201-source.tar"),
            "pages": 38,
        },
        "repository": repository_inventory(repo),
        "zenodo": zenodo_audit(official),
        "gaussian": gaussian_audit(repo, source),
        "notebooks": notebook_audit(repo),
        "mutual_information": mi_audit(repo),
        "differential_abundance": differential_abundance_audit(repo),
        "batch_correction": batch_audit(repo),
        "treatment_response": treatment_audit(repo),
        "claims": [
            {"id": "C1", "score": 2, "verdict": "SUPPORTED_ANALYTIC_ORACLE_AND_RELEASED_ODE"},
            {"id": "C2", "score": 1, "verdict": "PARTIAL_SMALL_RUN_AND_CACHED_RESULTS_REPLICATE_MISMATCH"},
            {"id": "C3", "score": 1, "verdict": "PARTIAL_QUALITATIVE_RANKING_NUMERIC_NOTEBOOK_TABLE_MISMATCH"},
            {"id": "C4", "score": 1, "verdict": "PARTIAL_EXECUTED_METRICS_MATCH_TABLE_RAW_INPUTS_EXTERNAL"},
            {"id": "C5", "score": 1, "verdict": "PARTIAL_EXECUTED_SHAPES_AND_FIGURES_RAW_LLR_EXTERNAL"},
            {"id": "C6", "score": 1, "verdict": "PARTIAL_COMBOSCIPLEX_CORRELATION_PATIENT_RESULT_VISUAL_ONLY"},
        ],
        "prepared_score": 7,
        "score_denominator": 12,
    }
    if run_runtime:
        audit["runtime"] = runtime_checks(repo, include_small_training=small_training)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--output", type=Path, default=HERE / "evidence" / "artifact_audit.json")
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--small-training", action="store_true")
    args = parser.parse_args()
    audit = build_audit(args.official.resolve(), args.runtime, args.small_training)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
