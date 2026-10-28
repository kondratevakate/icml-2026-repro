from __future__ import annotations

import argparse
import importlib.util
import json
import math
import py_compile
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT / "official" / "repo"
SOURCE_ROOT = PACKAGE_ROOT / "official" / "source"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def released_prior_check(epsilon: float = 1e-9) -> dict[str, Any]:
    module = _load_module(
        "official_embedding_conditional_prior",
        REPO_ROOT / "method" / "embedding_conditional_prior.py",
    )
    global_params = {"w": torch.tensor([3.0, 4.0])}
    model = module.EmbeddingConditionalPrior(1, global_params, {}, "cpu")
    model.W_mu["w"].data[:] = torch.tensor([[1.0], [0.0]])

    def offset(z: float) -> torch.Tensor:
        return model(
            torch.tensor([z]), global_params=global_params, adaptation_scale=0.2
        )[0]["w"].detach()

    z0, zsmall, z1, z2 = offset(0.0), offset(epsilon), offset(1.0), offset(2.0)
    # Exact TV distance between N(0,1) and N(1,1).
    released_tv_jump = math.erf(0.5 / math.sqrt(2.0))
    paper_rhs = epsilon / 2.0
    return {
        "global_norm": float(global_params["w"].norm()),
        "adaptation_scale": 0.2,
        "offset_z0": z0.tolist(),
        "offset_z_epsilon": zsmall.tolist(),
        "offset_z1": z1.tolist(),
        "offset_z2": z2.tolist(),
        "scale_invariance_error": float((z1 - z2).abs().max()),
        "jump_norm_at_zero": float((zsmall - z0).norm()),
        "epsilon": epsilon,
        "exact_tv_jump_for_unit_covariance_gaussians": released_tv_jump,
        "paper_linear_lipschitz_rhs_M1_W1_sigma1": paper_rhs,
        "released_map_violates_linear_bound": released_tv_jump > paper_rhs,
    }


def paper_lipschitz_check() -> dict[str, Any]:
    """Numerically check the exact Gaussian-TV value against the paper bound."""
    distances = [0.0, 0.01, 0.1, 1.0, 3.0]
    rows = []
    for distance in distances:
        exact_tv = math.erf(distance / (2.0 * math.sqrt(2.0)))
        pinsker_bound = distance / 2.0
        rows.append(
            {
                "embedding_distance": distance,
                "exact_total_variation": exact_tv,
                "paper_bound_M1_W1_sigma1": pinsker_bound,
                "bound_holds": exact_tv <= pinsker_bound + 1e-15,
            }
        )
    return {
        "rows": rows,
        "all_bounds_hold": all(row["bound_holds"] for row in rows),
        "error_decomposition_is_triangle_inequality": True,
    }


def theorem_constant_check() -> dict[str, Any]:
    """Counterexample to dropping the condition number in Appendix Theorem 5."""
    kappa, kappa0 = 1.0, 0.5
    w = torch.diag(torch.tensor([10.0, 1.0], dtype=torch.float64))
    z = torch.tensor([0.0, 1.0], dtype=torch.float64)
    error = torch.tensor([0.1, 0.0], dtype=torch.float64)
    appendix_c = kappa0 / kappa
    proper_c = appendix_c / 10.0
    appendix_condition = float(error.norm()) <= appendix_c * float(z.norm())
    required_parameter_condition = (
        kappa * float((w @ error).norm())
        <= kappa0 * float((w @ z).norm())
    )
    return {
        "singular_values": [10.0, 1.0],
        "condition_number": 10.0,
        "kappa": kappa,
        "kappa0": kappa0,
        "z_norm": float(z.norm()),
        "embedding_error_norm": float(error.norm()),
        "appendix_constant_kappa0_over_kappa": appendix_c,
        "condition_number_aware_constant": proper_c,
        "appendix_embedding_condition_holds": appendix_condition,
        "required_parameter_condition_holds": required_parameter_condition,
        "counterexample": appendix_condition and not required_parameter_condition,
    }


def release_inventory() -> dict[str, Any]:
    tracked = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "ls-tree", "-r", "--name-only", "HEAD"],
        text=True,
        encoding="utf-8",
    ).splitlines()
    py_files = sorted(REPO_ROOT.rglob("*.py"))
    compile_errors: dict[str, str] = {}
    for path in py_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            compile_errors[str(path.relative_to(REPO_ROOT))] = str(exc)

    tabular = pd.read_csv(REPO_ROOT / "data" / "example" / "latent_tabular_data.csv")
    embeddings = pd.read_csv(
        REPO_ROOT / "data" / "example" / "latent_embeddings_causal_noise0.0.csv"
    )
    source = (SOURCE_ROOT / "main.tex").read_text(encoding="utf-8")
    prior_code = (
        REPO_ROOT / "method" / "embedding_conditional_prior.py"
    ).read_text(encoding="utf-8")
    main_code = (REPO_ROOT / "method" / "main.py").read_text(encoding="utf-8")
    return {
        "tracked_file_count": len(tracked),
        "python_file_count": len(py_files),
        "python_compile_errors": compile_errors,
        "example_tabular_rows": len(tabular),
        "example_embedding_rows": len(embeddings),
        "has_tests": any("test" in Path(path).name.lower() for path in tracked),
        "has_cached_metrics_or_checkpoints": any(
            (
                any(part.lower() in {"results", "outputs"} for part in Path(path).parts)
                or Path(path).suffix.lower() in {".pth", ".pt", ".ckpt"}
            )
            for path in tracked
        ),
        "has_ukbb_data": any("ukbb" in path.lower() for path in tracked),
        "paper_linear_prior": r"\mathcal{N}(\theta+Wz, \sigma^2I)" in source,
        "release_normalizes_adaptation": "target_norm = adaptation_scale * global_norm" in prior_code,
        "main_sets_torch_seed": "torch.manual_seed" in main_code,
        "main_sets_numpy_seed": "np.random.seed" in main_code,
    }


def smoke_observations() -> dict[str, Any]:
    return {
        "command_scope": (
            "Official sequence/adaptive toy path, one epoch, one MC sample, "
            "one inner update, seed argument 999, CPU."
        ),
        "attempt_1": {
            "validation_auroc": 0.7188,
            "outcome": "failed because the entrypoint did not create results/",
        },
        "attempt_2": {
            "validation_auroc": 0.7276,
            "average_test_auroc": 0.6160,
            "outcome": "metrics saved, then cp1252 failed on final emoji print",
        },
        "attempt_3_utf8": {
            "validation_auroc": 0.6531,
            "average_test_auroc": 0.5666037752884746,
            "average_test_auprc": 0.38805493598625074,
            "outcome": "completed with PYTHONUTF8=1",
        },
        "same_seed_validation_auroc_range": 0.7276 - 0.6531,
        "interpretation": (
            "The random_seed argument controls data splitting but method/main.py "
            "does not seed torch or NumPy, so training is not repeatable."
        ),
        "expert_smoke": {
            "mode": "BALD",
            "targets": 5,
            "queries_per_target": 2,
            "svi_steps": 20,
            "svi_steps_per_query": 20,
            "completed": True,
        },
    }


def run_audit() -> dict[str, Any]:
    return {
        "paper": {
            "title": (
                "Bayesian Meta-Learning with Expert Feedback for Task-Shift "
                "Adaptation through Causal Embeddings"
            ),
            "openreview_id": "k76ll7aQyE",
            "arxiv": "2602.19788v1",
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "released_prior": released_prior_check(),
        "paper_proposition": paper_lipschitz_check(),
        "theorem_constant": theorem_constant_check(),
        "release_inventory": release_inventory(),
        "toy_smoke": smoke_observations(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "evidence" / "audit.json",
    )
    args = parser.parse_args()
    result = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

