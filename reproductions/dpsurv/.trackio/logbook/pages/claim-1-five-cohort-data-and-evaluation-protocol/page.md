# Claim 1: five-cohort data and evaluation protocol


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ceeee40f1771", "created_at": "2026-07-30T08:55:39+00:00", "title": "Claim 1: five-cohort data and evaluation protocol"}
-->
**PARTIAL - 1/2.** All five cohorts and five folds are present.
Every case and slide is held out exactly once, with no outer train/test overlap.
Eighty-six unique rows lack a DSS endpoint and are filtered by the trainer.
WSI, UNI2-h, and PANTHER inputs are external.


---
<!-- trackio-cell
{"type": "code", "id": "cell_221b1d5434c0", "created_at": "2026-07-30T08:55:39+00:00", "title": "Claim 1: five-cohort data and evaluation protocol evidence", "language": "python"}
-->
````python title=audit_dpsurv.py
"""Deterministic independent checks for the official DPsurv release."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions.normal import Normal


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_splits(repo: Path) -> dict[str, Any]:
    split_root = repo / "data" / "splits"
    cohorts: dict[str, dict[str, Any]] = {}
    for cohort in ("BLCA", "BRCA", "KIRC", "LUAD", "UCEC"):
        fold_summaries = []
        test_case_counts: Counter[str] = Counter()
        test_slide_counts: Counter[str] = Counter()
        cohort_cases: set[str] = set()
        cohort_slides: set[str] = set()
        for fold in range(5):
            folder = split_root / f"TCGA_{cohort}_overall_survival_k={fold}"
            train = read_rows(folder / "train.csv")
            test = read_rows(folder / "test.csv")
            train_cases = {row["case_id"] for row in train}
            test_cases = {row["case_id"] for row in test}
            train_slides = {row["slide_id"] for row in train}
            test_slides = {row["slide_id"] for row in test}
            test_case_counts.update(test_cases)
            test_slide_counts.update(test_slides)
            cohort_cases |= train_cases | test_cases
            cohort_slides |= train_slides | test_slides
            required = {
                "case_id",
                "slide_id",
                "dss_censorship",
                "dss_survival_days",
            }
            columns = set(train[0]) if train else set()
            invalid_dss = sum(
                not row["dss_censorship"] or not row["dss_survival_days"]
                for row in train + test
            )
            fold_summaries.append(
                {
                    "fold": fold,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_cases": len(train_cases),
                    "test_cases": len(test_cases),
                    "case_overlap": len(train_cases & test_cases),
                    "slide_overlap": len(train_slides & test_slides),
                    "required_columns_present": required <= columns,
                    "missing_dss_values": invalid_dss,
                }
            )
        cohorts[cohort] = {
            "folds": fold_summaries,
            "unique_cases": len(cohort_cases),
            "unique_slides": len(cohort_slides),
            "each_case_once_in_test": bool(test_case_counts)
            and all(count == 1 for count in test_case_counts.values())
            and set(test_case_counts) == cohort_cases,
            "each_slide_once_in_test": bool(test_slide_counts)
            and all(count == 1 for count in test_slide_counts.values())
            and set(test_slide_counts) == cohort_slides,
        }
    all_folds_disjoint = all(
        fold["case_overlap"] == 0
        and fold["slide_overlap"] == 0
        and fold["required_columns_present"]
        for item in cohorts.values()
        for fold in item["folds"]
    )
    all_endpoint_rows_complete = all(
        fold["missing_dss_values"] == 0
        for item in cohorts.values()
        for fold in item["folds"]
    )
    return {
        "cohorts": cohorts,
        "cohort_count": len(cohorts),
        "fold_directory_count": len(list(split_root.glob("TCGA_*_k=*"))),
        "csv_count": len(list(split_root.rglob("*.csv"))),
        "all_train_test_folds_disjoint": all_folds_disjoint,
        "all_released_rows_have_dss_endpoint": all_endpoint_rows_complete,
        "all_cases_once_in_test": all(
            item["each_case_once_in_test"] for item in cohorts.values()
        ),
        "all_slides_once_in_test": all(
            item["each_slide_once_in_test"] for item in cohorts.values()
        ),
        "directory_endpoint_label": "overall_survival",
        "trainer_endpoint_columns": [
            "dss_survival_days",
            "dss_censorship",
        ],
    }


def make_prototypes(
    components: int = 2,
    prototypes: int = 2,
    mean_dim: int = 3,
) -> list[dict[str, torch.Tensor]]:
    result = []
    input_dim = 1 + 2 * mean_dim
    for component in range(components):
        w = torch.zeros(prototypes, mean_dim, dtype=torch.float64)
        for index in range(prototypes):
            w[index, (component + index) % mean_dim] = 1.0
        result.append(
            {
                "alpha": torch.tensor(
                    [1.5 + component, 2.0 + component], dtype=torch.float64
                ),
                "Beta": torch.zeros(prototypes, input_dim, dtype=torch.float64),
                "sig": torch.tensor([0.5, 0.7], dtype=torch.float64),
                "eta": torch.tensor([1.0, 1.2], dtype=torch.float64),
                "gam": torch.tensor([0.8, 1.1], dtype=torch.float64),
                "W": w,
            }
        )
    return result


def component_survival(
    output: dict[str, torch.Tensor],
    qbins: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return component-wise plausibility and belief survival curves."""
    mux_all = output["mux"].reshape(output["mux"].shape[0], -1)
    sig2x_all = output["sig2x"].reshape(output["sig2x"].shape[0], -1)
    hx_all = output["hx"].reshape(output["hx"].shape[0], -1)
    log_bins = torch.log(qbins).view(1, -1)
    plausibility = []
    belief = []
    for index in range(len(mux_all)):
        mux = mux_all[index].view(-1, 1)
        sig2x = sig2x_all[index].view(-1, 1)
        sigx = torch.sqrt(sig2x)
        hx = hx_all[index].view(-1, 1)
        z2 = hx * sig2x + 1
        z = torch.sqrt(z2)
        sig1 = sigx * z
        pl = 1 / z * torch.exp(-0.5 * hx * (log_bins - mux) ** 2 / z2)
        fy1 = Normal(mux, sigx).cdf(log_bins) - pl * Normal(
            mux, sig1
        ).cdf(log_bins)
        fy2 = fy1 + pl
        plausibility.append(1 - fy1)
        belief.append(1 - fy2)
    return torch.stack(plausibility), torch.stack(belief)


def synthetic_model_audit(repo: Path) -> dict[str, Any]:
    models = load_module(
        "official_dpsurv_models",
        repo / "downstream" / "dpsurv" / "models.py",
    )
    losses = load_module(
        "official_dpsurv_losses",
        repo / "downstream" / "dpsurv" / "losses.py",
    )
    torch.manual_seed(17)
    prototypes = make_prototypes()
    model = models.mixture_ENNreg_new(
        input_dim=7,
        prototype_list=prototypes,
        num_models=2,
    )
    model.reset_parameters(prototypes, torch.device("cpu"))
    batch = 4
    prob = torch.tensor(
        [[0.8, 0.2], [0.4, 0.6], [0.1, 0.9], [0.5, 0.5]],
        dtype=torch.float32,
    )
    means = torch.randn(batch, 2, 3)
    cov = torch.rand(batch, 2, 3) + 0.1
    features = torch.cat([prob.unsqueeze(2), means, cov], dim=2)
    output = model(features, prob)
    qbins = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float32)
    labels = torch.tensor([0, 1, 2, 0])
    censorship = torch.tensor([0, 1, 0, 1])
    loss_fn = losses.Mixture_Evidential_nll_Loss(
        qbins=qbins,
        lambd=0.5,
        xi=0.0,
        rho=0.0,
    )
    loss = loss_fn(output, labels, censorship, prob)["loss"]
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    plausibility, belief = component_survival(output, qbins)

    lambda_results = {}
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        official_components = value * plausibility + value * belief
        paper_components = value * belief + (1.0 - value) * plausibility
        evaluation_components = (
            value * plausibility + (1.0 - value) * belief
        )
        official_curve = torch.sum(
            prob.T[:, :, None] * official_components, dim=0
        )
        paper_curve = torch.sum(prob.T[:, :, None] * paper_components, dim=0)
        evaluation_curve = torch.sum(
            prob.T[:, :, None] * evaluation_components, dim=0
        )
        lambda_results[str(value)] = {
            "training_vs_paper_max_abs": float(
                torch.max(torch.abs(official_curve - paper_curve)).detach()
            ),
            "evaluation_vs_paper_max_abs": float(
                torch.max(torch.abs(evaluation_curve - paper_curve)).detach()
            ),
            "official_training_curve_max": float(
                official_curve.max().detach()
            ),
            "paper_curve_max": float(paper_curve.max().detach()),
        }

    low_prob_error = None
    empty_init = models.ENNreg_init_cosine(
        torch.zeros(4),
        torch.randn(4, 3),
        torch.log(torch.tensor([2.0, 3.0, 4.0, 5.0])),
        K=2,
        prob_thresh=0.1,
    )
    try:
        empty_model = models.mixture_ENNreg_new(
            input_dim=7,
            prototype_list=[empty_init],
            num_models=1,
        )
        empty_model.reset_parameters([empty_init], torch.device("cpu"))
        empty_model(torch.randn(2, 1, 7), torch.ones(2, 1))
    except Exception as error:  # independent reproduction of release behavior
        low_prob_error = f"{type(error).__name__}: {error}"

    return {
        "output_shapes": {
            key: list(value.shape) for key, value in output.items()
        },
        "forward_finite": all(
            bool(torch.isfinite(value).all()) for value in output.values()
        ),
        "loss": float(loss.detach()),
        "loss_finite": bool(torch.isfinite(loss)),
        "finite_gradient_tensors": sum(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        ),
        "gradient_tensor_count": len(gradients),
        "belief_never_exceeds_plausibility": bool(
            torch.all(belief <= plausibility + 1e-7)
        ),
        "component_mixture_weights_sum_to_one": bool(
            torch.allclose(prob.sum(dim=1), torch.ones(batch))
        ),
        "lambda_audit": lambda_results,
        "lambda_interpretation": (
            "The paper uses lambda*Bel+(1-lambda)*Pl. Training uses "
            "lambda*Pl+lambda*Bel; evaluation uses "
            "lambda*Pl+(1-lambda)*Bel, reversing the stated direction. "
            "Both coincide with the paper only at lambda=0.5."
        ),
        "all_low_probability_component": {
            "selected_prototype_count": int(empty_init["W"].shape[0]),
            "error": low_prob_error,
        },
    }


def release_inventory(repo: Path) -> dict[str, Any]:
    empirical_suffixes = {
        ".pt",
        ".pth",
        ".ckpt",
        ".pkl",
        ".npy",
        ".npz",
        ".json",
        ".log",
    }
    empirical = [
        str(path.relative_to(repo))
        for path in repo.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and path.suffix.lower() in empirical_suffixes
    ]
    notebook_outputs = {}
    for path in repo.rglob("*.ipynb"):
        data = json.loads(path.read_text(encoding="utf-8"))
        notebook_outputs[str(path.relative_to(repo))] = {
            "cells": len(data.get("cells", [])),
            "output_objects": sum(
                len(cell.get("outputs", [])) for cell in data.get("cells", [])
            ),
            "executed_cells": sum(
                cell.get("execution_count") is not None
                for cell in data.get("cells", [])
            ),
        }
    return {
        "python_files": len(list(repo.rglob("*.py"))),
        "split_csv_files": len(list((repo / "data" / "splits").rglob("*.csv"))),
        "empirical_binary_or_metric_artifacts": empirical,
        "notebook_outputs": notebook_outputs,
        "external_inputs_required": [
            "TCGA whole-slide images",
            "UNI2-h patch features",
            "PANTHER GMM embeddings",
        ],
    }


def visualization_audit(repo: Path) -> dict[str, Any]:
    path = repo / "visualization" / "prototype_visualization_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    encoder_definitions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "get_panther_encoder"
    ]
    sys.path.insert(0, str(repo))
    module = load_module("official_dpsurv_visualization", path)
    error = None
    try:
        module.get_panther_encoder(3, 2, "missing.pkl")
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
    return {
        "get_panther_encoder_definition_count": len(encoder_definitions),
        "effective_entrypoint_error": error,
        "interpretation": (
            "The second get_panther_encoder definition shadows the first and "
            "calls create_embedding_model, which is neither defined nor "
            "imported in the release."
        ),
    }


def audit(
    repo: Path,
    paper: Path,
    source_archive: Path,
) -> dict[str, Any]:
    split = audit_splits(repo)
    model = synthetic_model_audit(repo)
    release = release_inventory(repo)
    visualization = visualization_audit(repo)
    checks = {
        "five_cohorts_and_five_folds_present": split["cohort_count"] == 5
        and split["fold_directory_count"] == 25
        and split["csv_count"] == 50,
        "train_test_splits_are_disjoint": split[
            "all_train_test_folds_disjoint"
        ],
        "all_released_rows_have_dss_endpoint": split[
            "all_released_rows_have_dss_endpoint"
        ],
        "each_case_appears_once_in_test": split["all_cases_once_in_test"],
        "synthetic_forward_and_backward_execute": model["forward_finite"]
        and model["loss_finite"]
        and model["finite_gradient_tensors"] == model["gradient_tensor_count"],
        "belief_is_bounded_by_plausibility": model[
            "belief_never_exceeds_plausibility"
        ],
        "training_matches_paper_at_lambda_half": model["lambda_audit"]["0.5"][
            "training_vs_paper_max_abs"
        ]
        < 1e-6,
        "training_matches_paper_away_from_lambda_half": model["lambda_audit"][
            "0.25"
        ]["training_vs_paper_max_abs"]
        < 1e-6,
        "evaluation_matches_paper_away_from_lambda_half": model[
            "lambda_audit"
        ]["0.25"]["evaluation_vs_paper_max_abs"]
        < 1e-6,
        "all_low_probability_component_is_supported": model[
            "all_low_probability_component"
        ]["error"]
        is None,
        "run_level_empirical_artifacts_present": bool(
            release["empirical_binary_or_metric_artifacts"]
        ),
        "visualization_entrypoint_executes": visualization[
            "effective_entrypoint_error"
        ]
        is None,
    }
    return {
        "artifact_hashes": {
            "paper_sha256": sha256(paper),
            "source_sha256": sha256(source_archive),
        },
        "checks": checks,
        "passed": sum(checks.values()),
        "failed": len(checks) - sum(checks.values()),
        "splits": split,
        "model_and_theory": model,
        "release": release,
        "visualization": visualization,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--repo", type=Path, default=here / "official" / "repo")
    parser.add_argument(
        "--paper",
        type=Path,
        default=here / "official" / "paper.pdf",
    )
    parser.add_argument(
        "--source-archive",
        type=Path,
        default=here / "official" / "source.tar.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "evidence" / "audit_results.json",
    )
    args = parser.parse_args()
    result = audit(args.repo, args.paper, args.source_archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

````


````output
{
  "cohorts": {
    "BLCA": {
      "folds": [
        {
          "fold": 0,
          "train_rows": 356,
          "test_rows": 81,
          "train_cases": 299,
          "test_cases": 74,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 13
        },
        {
          "fold": 1,
          "train_rows": 365,
          "test_rows": 72,
          "train_cases": 301,
          "test_cases": 72,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 13
        },
        {
          "fold": 2,
          "train_rows": 314,
          "test_rows": 123,
          "train_cases": 294,
          "test_cases": 79,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 13
        },
        {
          "fold": 3,
          "train_rows": 351,
          "test_rows": 86,
          "train_cases": 297,
          "test_cases": 76,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 13
        },
        {
          "fold": 4,
          "train_rows": 362,
          "test_rows": 75,
          "train_cases": 301,
          "test_cases": 72,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 13
        }
      ],
      "unique_cases": 373,
      "unique_slides": 437,
      "each_case_once_in_test": true,
      "each_slide_once_in_test": true
    },
    "BRCA": {
      "folds": [
        {
          "fold": 0,
          "train_rows": 904,
          "test_rows": 207,
          "train_cases": 837,
          "test_cases": 204,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 20
        },
        {
          "fold": 1,
          "train_rows": 870,
          "test_rows": 241,
          "train_cases": 819,
          "test_cases": 222,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 20
        },
        {
          "fold": 2,
          "train_rows": 894,
          "test_rows": 217,
          "train_cases": 828,
          "test_cases": 213,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 20
        },
        {
          "fold": 3,
          "train_rows": 931,
          "test_rows": 180,
          "train_cases": 861,
          "test_cases": 180,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 20
        },
        {
          "fold": 4,
          "train_rows": 845,
          "test_rows": 266,
          "train_cases": 819,
          "test_cases": 222,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 20
        }
      ],
      "unique_cases": 1041,
      "unique_slides": 1111,
      "each_case_once_in_test": true,
      "each_slide_once_in_test": true
    },
    "KIRC": {
      "folds": [
        {
          "fold": 0,
          "train_rows": 458,
          "test_rows": 59,
          "train_cases": 453,
          "test_cases": 58,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 9
        },
        {
          "fold": 1,
          "train_rows": 433,
          "test_rows": 84,
          "train_cases": 431,
          "test_cases": 80,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 9
        },
        {
          "fold": 2,
          "train_rows": 377,
          "test_rows": 140,
          "train_cases": 371,
          "test_cases": 140,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 9
        },
        {
          "fold": 3,
          "train_rows": 421,
          "test_rows": 96,
          "train_cases": 416,
          "test_cases": 95,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 9
        },
        {
          "fold": 4,
          "train_rows": 379,
          "test_rows": 138,
          "train_cases": 373,
          "test_cases": 138,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 9
        }
      ],
      "unique_cases": 511,
      "unique_slides": 517,
      "each_case_once_in_test": true,
      "each_slide_once_in_test": true
    },
    "LUAD": {
      "folds": [
        {
          "fold": 0,
          "train_rows": 364,
          "test_rows": 155,
          "train_cases": 359,
          "test_cases": 97,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 44
        },
        {
          "fold": 1,
          "train_rows": 387,
          "test_rows": 132,
          "train_cases": 325,
          "test_cases": 131,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 44
        },
        {
          "fold": 2,
          "train_rows": 409,
          "test_rows": 110,
          "train_cases": 346,
          "test_cases": 110,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 44
        },
        {
          "fold": 3,
          "train_rows": 425,
          "test_rows": 94,
          "train_cases": 366,
          "test_cases": 90,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 44
        },
        {
          "fold": 4,
          "train_rows": 491,
          "test_rows": 28,
          "train_cases": 428,
          "test_cases": 28,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 44
        }
      ],
      "unique_cases": 456,
      "unique_slides": 519,
      "each_case_once_in_test": true,
      "each_slide_once_in_test": true
    },
    "UCEC": {
      "folds": [
        {
          "fold": 0,
          "train_rows": 444,
          "test_rows": 121,
          "train_cases": 415,
          "test_cases": 89,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 0
        },
        {
          "fold": 1,
          "train_rows": 439,
          "test_rows": 126,
          "train_cases": 406,
          "test_cases": 98,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 0
        },
        {
          "fold": 2,
          "train_rows": 440,
          "test_rows": 125,
          "train_cases": 379,
          "test_cases": 125,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 0
        },
        {
          "fold": 3,
          "train_rows": 464,
          "test_rows": 101,
          "train_cases": 403,
          "test_cases": 101,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 0
        },
        {
          "fold": 4,
          "train_rows": 473,
          "test_rows": 92,
          "train_cases": 413,
          "test_cases": 91,
          "case_overlap": 0,
          "slide_overlap": 0,
          "required_columns_present": true,
          "missing_dss_values": 0
        }
      ],
      "unique_cases": 504,
      "unique_slides": 565,
      "each_case_once_in_test": true,
      "each_slide_once_in_test": true
    }
  },
  "cohort_count": 5,
  "fold_directory_count": 25,
  "csv_count": 50,
  "all_train_test_folds_disjoint": true,
  "all_released_rows_have_dss_endpoint": false,
  "all_cases_once_in_test": true,
  "all_slides_once_in_test": true,
  "directory_endpoint_label": "overall_survival",
  "trainer_endpoint_columns": [
    "dss_survival_days",
    "dss_censorship"
  ]
}
````
