#!/usr/bin/env python3
"""Independent release audit for CAML (arXiv:2605.25001)."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
OFFICIAL = CANDIDATE / "official"
REPO = OFFICIAL / "repo"
TEX = OFFICIAL / "source" / "example_paper.tex"

EXPECTED_PAPER_SHA256 = (
    "4ca6809002a0ab786781b53e30dc7c200c7a1bf7b9acee676d1cf09c8af6152c"
)
EXPECTED_SOURCE_SHA256 = (
    "13c19394c76c86f337f8fe37dd538134fbfd93c5b8b6aee4d64efc6cdedd1646"
)
EXPECTED_COMMIT = "8f4daabf0db60f4f185a1a9b791ad7d7be41a033"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args],
        text=True,
        encoding="utf-8",
    ).strip()


def artifact_audit() -> dict[str, Any]:
    paper = OFFICIAL / "paper.pdf"
    source = OFFICIAL / "source.tar.gz"
    tracked = git("ls-files").splitlines()
    return {
        "paper": {
            "bytes": paper.stat().st_size,
            "sha256": sha256(paper),
            "expected_sha256": EXPECTED_PAPER_SHA256,
            "matches": sha256(paper) == EXPECTED_PAPER_SHA256,
        },
        "source": {
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
            "expected_sha256": EXPECTED_SOURCE_SHA256,
            "matches": sha256(source) == EXPECTED_SOURCE_SHA256,
        },
        "repository": {
            "commit": git("rev-parse", "HEAD"),
            "expected_commit": EXPECTED_COMMIT,
            "commit_matches": git("rev-parse", "HEAD") == EXPECTED_COMMIT,
            "tracked_files": len(tracked),
            "result_files": [
                name
                for name in tracked
                if Path(name).suffix.lower() in {".csv", ".json", ".npy", ".npz", ".pt", ".pth"}
            ],
            "test_files": [
                name for name in tracked if "test" in Path(name).name.lower()
            ],
            "requirements_file": "requirements.txt" in tracked,
        },
    }


def theorem_counterexample() -> dict[str, Any]:
    """Refute 'non-unique implies connected/non-isolated' with N(u)=u^2."""
    roots = np.array([-1.0, 1.0])
    residuals = roots**2 - 1.0
    derivative_at_roots = 2.0 * roots
    return {
        "operator": "N(u) = u^2",
        "rhs": 1.0,
        "solutions": roots.tolist(),
        "residuals": residuals.tolist(),
        "non_unique": len(roots) > 1,
        "minimum_pairwise_distance": float(abs(roots[1] - roots[0])),
        "solutions_are_isolated": bool(np.all(np.abs(derivative_at_roots) > 0)),
        "solution_set_connected": False,
        "headline_implication_valid": False,
        "missing_nonlinear_assumptions": [
            "non-trivial kernel of DN[u*]",
            "constant-rank or regular-value condition",
            "connected parameter domain for a continuous solution family",
            "representability of that family by the neural network",
        ],
    }


def _load_official_caml():
    sys.path.insert(0, str(REPO))
    try:
        return importlib.import_module("caml").CAML
    finally:
        if sys.path[0] == str(REPO):
            sys.path.pop(0)


class LinearPDE:
    def __init__(self, gamma: torch.Tensor, target: torch.Tensor):
        self.gamma = gamma
        self.target = target

    def __call__(self, mesh: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        del mesh
        return self.gamma * pred - self.target

    def reset(self) -> None:
        return None


class LinearBC:
    def __init__(self, alpha: torch.Tensor, target: torch.Tensor):
        self.alpha = alpha
        self.target = target

    def __call__(
        self, mesh: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        del mesh
        return self.alpha * pred[mask] - self.target

    def reset(self) -> None:
        return None


def aligned_constraint_check(seed: int = 260525001) -> dict[str, Any]:
    torch.manual_seed(seed)
    dtype = torch.float64
    pred = torch.tensor([[0.2], [-0.7], [1.1], [0.4]], dtype=dtype, requires_grad=True)
    mesh = torch.zeros((4, 1), dtype=dtype)
    mask = torch.tensor([False, False, True, True])
    gamma = torch.tensor([[2.0], [-1.0], [0.5], [1.5]], dtype=dtype)
    pde_target = torch.tensor([[0.3], [0.2], [-0.4], [0.6]], dtype=dtype)
    alpha = torch.tensor([[1.2], [-0.8]], dtype=dtype)
    bc_target = torch.tensor([[0.1], [-0.2]], dtype=dtype)
    w_res, w_bc = 1.7, 2.3
    pde = LinearPDE(gamma, pde_target)
    bc = LinearBC(alpha, bc_target)
    CAML = _load_official_caml()
    module = CAML(
        mask=mask,
        pde=pde,
        boundary_condition=bc,
        w_res=w_res,
        w_bc=w_bc,
        td=2,
        tr=3,
        linear=True,
    )

    r = gamma * pred.detach() - pde_target
    s = alpha * pred.detach()[mask] - bc_target
    expected_c = -(
        w_res / len(r) * (gamma * r).sum()
        + w_bc / len(s) * (alpha * s).sum()
    ) / (
        w_res / len(r) * (gamma**2).sum()
        + w_bc / len(s) * (alpha**2).sum()
    )
    output = module(mesh=mesh, pred=pred, mask=mask, t=3)
    actual_c = output["c"].reshape(())
    schedule = {str(t): module._get_lambda(t) for t in (0, 1, 2, 3, 4, 5, 6)}
    return {
        "seed": seed,
        "expected_c": float(expected_c),
        "official_c": float(actual_c),
        "absolute_difference": float(abs(actual_c - expected_c)),
        "offset_detached": not actual_c.requires_grad,
        "schedule": schedule,
        "expected_schedule": {
            "0": 0.0,
            "1": 0.0,
            "2": 0.0,
            "3": 1.0 / 3.0,
            "4": 2.0 / 3.0,
            "5": 1.0,
            "6": 1.0,
        },
        "total_is_finite": bool(torch.isfinite(output["total"])),
    }


def degenerate_offset_check() -> dict[str, Any]:
    CAML = _load_official_caml()
    pred = torch.zeros((2, 1), dtype=torch.float64)
    mesh = torch.zeros_like(pred)
    mask = torch.tensor([False, True])
    pde = LinearPDE(torch.zeros_like(pred), torch.ones_like(pred))
    bc = LinearBC(torch.zeros((1, 1), dtype=torch.float64), torch.ones((1, 1), dtype=torch.float64))
    module = CAML(mask, pde, bc, linear=True)
    output = module(mesh, pred, mask, t=0)
    value = float(output["c"].reshape(()))
    return {
        "case": "all zeroth-order coefficients are zero",
        "official_c": value,
        "is_finite": bool(np.isfinite(value)),
        "denominator_guard_present": False,
    }


def parse_assignments(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    names = (
        "backbone", "loss", "lr", "w_res", "w_bc", "t_d", "t_r",
        "min_epochs", "max_epochs", "target_l2",
    )
    values: dict[str, str] = {}
    for name in names:
        match = re.search(rf"^{name}\s*=\s*(.+?)\s*$", text, flags=re.MULTILINE)
        values[name] = match.group(1) if match else ""
    return values


def release_conformance() -> dict[str, Any]:
    scripts = {
        name: parse_assignments(REPO / f"eval_{name}.py")
        for name in ("heat", "poisson", "ns", "helm")
    }
    paper = {
        "heat": {"w_bc": "5", "min_epochs": "6000", "target_l2": "2.0e-3"},
        "poisson": {"w_bc": "100", "min_epochs": "6000", "target_l2": "1.0e-2"},
        "ns": {"w_bc": "100", "min_epochs": "6000", "target_l2": "5.0e-3"},
        "helm": {"w_bc": "10", "min_epochs": "4000", "target_l2": "1.0e-3"},
    }
    mismatches: list[dict[str, str]] = []
    for benchmark, expected in paper.items():
        for key, paper_value in expected.items():
            code_value = scripts[benchmark][key]
            normalized_code = code_value.lower().replace(".0", "")
            normalized_paper = paper_value.lower().replace(".0", "")
            try:
                equal = float(code_value) == float(paper_value)
            except ValueError:
                equal = normalized_code == normalized_paper
            if not equal:
                mismatches.append(
                    {
                        "benchmark": benchmark,
                        "parameter": key,
                        "paper": paper_value,
                        "code": code_value,
                    }
                )
    readme = (REPO / "readme.md").read_text(encoding="utf-8")
    imports_overrides = any(
        "from overrides import overrides" in path.read_text(encoding="utf-8")
        for path in (REPO / "benchmark").rglob("*.py")
    )
    return {
        "script_assignments": scripts,
        "paper_mlp_subset": paper,
        "mismatches": mismatches,
        "mismatch_count": len(mismatches),
        "readme_claims_only_standard_torch": (
            "does not require any additional specialized dependencies beyond the standard PyTorch" in readme
        ),
        "undocumented_overrides_import": imports_overrides,
        "baseline_loss_implementations_released": ["pinn", "caml"],
        "paper_baseline_loss_count": 6,
        "cached_empirical_outputs_released": False,
    }


def mini_heat_run(epochs: int = 10) -> dict[str, Any]:
    sys.path.insert(0, str(REPO))
    try:
        module = importlib.import_module("eval_heat")
        start = time.perf_counter()
        result = module.run_one_seed(
            999,
            min_epochs=epochs,
            max_epochs=epochs,
            target_l2=0.0,
        )
        elapsed = time.perf_counter() - start
    finally:
        if sys.path[0] == str(REPO):
            sys.path.pop(0)
    return {
        "seed": 999,
        "epochs": epochs,
        "l2_at_last_epoch": result[0],
        "reached_target_epoch": result[2],
        "positive_cosine_rate": result[3],
        "wall_seconds": elapsed,
        "canonical_table_reproduction": False,
    }


def run(include_mini_heat: bool = False, mini_epochs: int = 10) -> dict[str, Any]:
    claims = [
        {"id": "C1", "score": 0, "max_score": 2},
        {"id": "C2", "score": 2, "max_score": 2},
        {"id": "C3", "score": 1, "max_score": 2},
        {"id": "C4", "score": 0, "max_score": 2},
        {"id": "C5", "score": 0, "max_score": 2},
        {"id": "C6", "score": 1, "max_score": 2},
    ]
    result: dict[str, Any] = {
        "paper": {
            "title": "Mitigating Gradient Pathology in PINNs through Aligned Constraint",
            "openreview_id": "Fisw2kc7EY",
            "arxiv_id": "2605.25001v1",
        },
        "artifacts": artifact_audit(),
        "theorem_counterexample": theorem_counterexample(),
        "aligned_constraint": aligned_constraint_check(),
        "degenerate_offset": degenerate_offset_check(),
        "release_conformance": release_conformance(),
        "claims": claims,
        "prepared_score": sum(item["score"] for item in claims),
        "score_denominator": sum(item["max_score"] for item in claims),
        "limits": {
            "canonical_gpu": "NVIDIA RTX 5090",
            "canonical_five_seed_runs_completed": False,
            "full_table_regenerated": False,
        },
    }
    if include_mini_heat:
        result["mini_heat"] = mini_heat_run(mini_epochs)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-mini-heat", action="store_true")
    parser.add_argument("--mini-epochs", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=HERE / "evidence" / "audit.json"
    )
    args = parser.parse_args()
    result = run(args.include_mini_heat, args.mini_epochs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
