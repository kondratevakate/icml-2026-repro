# Claim 1: three-stage optimizer mechanics


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_14e86b17bd67", "created_at": "2026-07-29T17:12:57+00:00", "title": "Claim 1: three-stage optimizer mechanics"}
-->
**VERIFIED AT EQUATION LEVEL WITH QUALIFICATIONS — 2/2.**

Across 500 random problems, the maximum trust-region
boundary residual is
`4.441e-16`, the Stage-2 norm
relative error is `1.898e-12`,
and the fusion residual is zero.

The printed method has two edge qualifications. Equation 10 is undefined when
the optimized dual gradient is zero; symmetric opposite gradients expose this
`0/0` case. The method calls Stage 3 “adaptive,” but Algorithm 1 and equation
13 use a fixed tuned `nu` and supply no data-dependent adaptation rule.


---
<!-- trackio-cell
{"type": "code", "id": "cell_2f5bcf77a98f", "created_at": "2026-07-29T17:12:57+00:00", "title": "C1 machine-readable evidence", "language": "python"}
-->
````python title=came_grad_equations.py
"""Independent implementation of CAME-Grad equations 8-14.

This is not copied from, and cannot substitute for, the withheld official
optimizer. It exists solely to test consequences of the printed equations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class CameGradResult:
    alpha: np.ndarray
    joint: np.ndarray
    mean: np.ndarray
    dual_gradient: np.ndarray
    rectified: np.ndarray
    enhanced: np.ndarray
    final: np.ndarray
    dual_objective: float


def _validate(
    task_gradients: np.ndarray,
    task_weights: np.ndarray,
    rho: float,
    kappa: float,
    nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    gradients = np.asarray(task_gradients, dtype=np.float64)
    weights = np.asarray(task_weights, dtype=np.float64)
    if gradients.ndim != 2 or gradients.shape[0] < 1:
        raise ValueError("task_gradients must have shape (tasks, parameters)")
    if weights.shape != (gradients.shape[0],):
        raise ValueError("task_weights must have one value per task")
    if not np.isfinite(gradients).all() or not np.isfinite(weights).all():
        raise ValueError("inputs must be finite")
    if not 0 <= rho < 1:
        raise ValueError("rho must be in [0, 1)")
    if kappa < 1:
        raise ValueError("kappa must be at least 1")
    if not 0 <= nu <= 1:
        raise ValueError("nu must be in [0, 1]")
    return gradients, weights


def combine_gradients(
    task_gradients: np.ndarray,
    task_weights: np.ndarray,
    *,
    rho: float = 0.5,
    kappa: float = 1.5,
    nu: float = 0.2,
    epsilon: float = 1e-12,
) -> CameGradResult:
    """Apply the paper's three printed stages to a matrix of task gradients.

    A zero dual-gradient makes equation 10 undefined. This implementation
    raises instead of silently inventing an author behavior.
    """

    gradients, weights = _validate(
        task_gradients, task_weights, rho, kappa, nu
    )
    joint = weights @ gradients
    mean = gradients.mean(axis=0)
    radius = rho * np.linalg.norm(mean)

    def objective(alpha: np.ndarray) -> float:
        dual_gradient = alpha @ gradients
        return float(
            dual_gradient @ mean + radius * np.linalg.norm(dual_gradient)
        )

    tasks = gradients.shape[0]
    optimization = minimize(
        objective,
        np.full(tasks, 1.0 / tasks),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * tasks,
        constraints=[
            {
                "type": "eq",
                "fun": lambda alpha: float(np.sum(alpha) - 1.0),
            }
        ],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not optimization.success:
        raise RuntimeError(f"dual optimization failed: {optimization.message}")

    alpha = np.asarray(optimization.x, dtype=np.float64)
    dual_gradient = alpha @ gradients
    dual_norm = np.linalg.norm(dual_gradient)
    if radius > 0 and dual_norm <= epsilon:
        raise ZeroDivisionError(
            "paper equation 10 is undefined when g_alpha has zero norm"
        )
    if radius == 0:
        rectified = mean.copy()
    else:
        rectified = mean + radius * dual_gradient / dual_norm

    target_magnitude = kappa * np.linalg.norm(joint)
    rectified_norm = np.linalg.norm(rectified)
    enhanced = rectified * target_magnitude / (rectified_norm + epsilon)
    final = (1.0 - nu) * enhanced + nu * (kappa * joint)

    return CameGradResult(
        alpha=alpha,
        joint=joint,
        mean=mean,
        dual_gradient=dual_gradient,
        rectified=rectified,
        enhanced=enhanced,
        final=final,
        dual_objective=objective(alpha),
    )

````


````python title=audit_claims.py
#!/usr/bin/env python3
"""Deterministic CAME-Grad paper-equation and official-release audit."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from came_grad_equations import combine_gradients


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_anchors(tex_path: Path) -> dict:
    text = tex_path.read_text(encoding="utf-8")
    anchors = {
        "algorithm_three_stages": all(
            phrase in text
            for phrase in (
                "Conflict-Averse Direction Rectification",
                "Magnitude-Enhanced Energy Injection",
                "Adaptive Gradient Fusion",
            )
        ),
        "mimic_average_2_3": "2.3\\%" in text,
        "iu_average_1_9": "1.9\\%" in text,
        "trust_region_equation": r"\|\mathbf{u} - \boldsymbol{\mu}\| \le \rho \|\boldsymbol{\mu}\|" in text,
        "fixed_fusion_equation": r"\mathbf{g}_{final} = (1 - \nu) \mathbf{u}_{en} + \nu \mathbf{g}'_{joint}" in text,
    }
    # In Algorithm 1 and equation 13, nu is used directly. There is no update
    # or data-dependent definition that would make it adaptive.
    adaptive_nu_patterns = [
        r"\\nu_\{t",
        r"\\nu\s*\\leftarrow",
        r"\\nu\s*=" + r".*(?:conflict|cosine|iteration|epoch)",
    ]
    anchors["data_dependent_nu_rule_found"] = any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in adaptive_nu_patterns
    )
    def table_ce_differences(label: str) -> dict:
        label_token = rf"\label{{{label}}}"
        start = text.index(label_token)
        end = text.index(r"\end{table*}", start)
        block = text[start:end]
        rows = []
        for line in block.splitlines():
            if "&" not in line or not line.rstrip().endswith(r"\\"):
                continue
            values = [float(value) for value in re.findall(r"0\.\d+", line)]
            if len(values) < 4:
                continue
            name = line.split("&", 1)[0].strip()
            rows.append((name, values[-1]))
        pairs = []
        for index in range(0, len(rows), 2):
            baseline, came = rows[index : index + 2]
            if "CAME-Grad" not in came[0]:
                raise ValueError(f"unexpected row pairing in {label}: {baseline}, {came}")
            pairs.append(
                {
                    "model": baseline[0],
                    "baseline_ce_average": baseline[1],
                    "came_grad_ce_average": came[1],
                    "absolute_difference": came[1] - baseline[1],
                }
            )
        return {
            "pair_count": len(pairs),
            "pairs": pairs,
            "mean_absolute_difference": float(
                np.mean([pair["absolute_difference"] for pair in pairs])
            ),
        }

    return {
        "sha256": sha256(tex_path),
        "anchors": anchors,
        "published_table_arithmetic": {
            "mimic_cxr": table_ce_differences("tab:mimic_main"),
            "iu_xray": table_ce_differences("tab:iu_main"),
            "interpretation": (
                "Arithmetic recomputed from published TeX cells; this is "
                "source concordance, not experimental reproduction."
            ),
        },
    }


def release_audit(code_root: Path) -> dict:
    required = {
        "optimizer": code_root / "modules" / "CAME_Grad.py",
        "dataset_module": code_root / "dataset.py",
        "dataset_package": code_root / "dataset",
        "requirements": code_root / "requirements.txt",
    }
    trainer = code_root / "modules" / "trainer.py"
    train = code_root / "main_train.py"
    readme = code_root / "README.md"
    trainer_text = trainer.read_text(encoding="utf-8")
    train_text = train.read_text(encoding="utf-8")
    readme_text = readme.read_text(encoding="utf-8")

    python_files = sorted(code_root.rglob("*.py"))
    parse_errors = {}
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            parse_errors[path.relative_to(code_root).as_posix()] = str(exc)

    return {
        "file_count": sum(1 for path in code_root.rglob("*") if path.is_file() and ".git" not in path.parts),
        "python_file_count": len(python_files),
        "required_artifacts_present": {
            name: path.exists() for name, path in required.items()
        },
        "trainer_imports_missing_optimizer": (
            "from .CAME_Grad import CAME_Grad" in trainer_text
            and not required["optimizer"].exists()
        ),
        "train_imports_missing_dataset": (
            "from dataset import create_dataset" in train_text
            and not required["dataset_module"].exists()
            and not required["dataset_package"].exists()
        ),
        "chexbert_placeholder_present": "'xxxxxxxx'" in trainer_text,
        "readme_says_core_withheld": (
            "core optimizer source code is temporarily withheld" in readme_text
        ),
        "python_parse_errors": parse_errors,
        "files": {
            path.relative_to(code_root).as_posix(): sha256(path)
            for path in python_files
        },
    }


def mechanics_audit(seed: int = 260522635, repeats: int = 500) -> dict:
    rng = np.random.default_rng(seed)
    trust_residuals = []
    magnitude_relative_errors = []
    fusion_residuals = []
    simplex_residuals = []

    for _ in range(repeats):
        gradients = rng.normal(size=(3, 7))
        weights = rng.uniform(0.1, 2.0, size=3)
        rho = rng.uniform(0.01, 0.95)
        kappa = rng.uniform(1.0, 2.0)
        nu = rng.uniform(0.0, 1.0)
        result = combine_gradients(
            gradients, weights, rho=rho, kappa=kappa, nu=nu
        )
        radius = rho * np.linalg.norm(result.mean)
        trust_residuals.append(
            abs(np.linalg.norm(result.rectified - result.mean) - radius)
        )
        target = kappa * np.linalg.norm(result.joint)
        magnitude_relative_errors.append(
            abs(np.linalg.norm(result.enhanced) - target) / max(target, 1e-15)
        )
        expected_final = (1 - nu) * result.enhanced + nu * (
            kappa * result.joint
        )
        fusion_residuals.append(np.linalg.norm(result.final - expected_final))
        simplex_residuals.append(abs(result.alpha.sum() - 1.0))

    # Covariance counterexample: identical output norms do not determine
    # diffusion covariance. Distribution A is constant; distribution B flips
    # an orthogonal component while retaining the same norm.
    constant = np.repeat([[1.0, 0.0]], 20000, axis=0)
    signs = np.tile([-1.0, 1.0], 10000)
    variable = np.column_stack(
        [np.full(signs.shape, np.sqrt(0.5)), signs * np.sqrt(0.5)]
    )
    norm_a = np.linalg.norm(constant, axis=1)
    norm_b = np.linalg.norm(variable, axis=1)
    covariance_a = np.cov(constant, rowvar=False, bias=True)
    covariance_b = np.cov(variable, rowvar=False, bias=True)

    # With nu=1 and kappa=1, equation 13 exactly recovers the joint gradient.
    gradients = np.array([[1.0, 2.0], [-0.5, 1.5], [0.2, -0.1]])
    weights = np.array([1.0, 2.0, 0.5])
    identity = combine_gradients(
        gradients, weights, rho=0.5, kappa=1.0, nu=1.0
    )

    # A symmetric conflict makes g_alpha exactly zero for alpha=(0.5, 0.5),
    # exposing equation 10's unspecified 0/0 degeneracy when mu is also zero.
    symmetric = np.array([[1.0, 0.0], [-1.0, 0.0]])
    symmetric_mean = symmetric.mean(axis=0)
    symmetric_dual = np.array([0.5, 0.5]) @ symmetric

    return {
        "seed": seed,
        "repeats": repeats,
        "max_trust_region_boundary_residual": max(trust_residuals),
        "max_stage2_magnitude_relative_error": max(magnitude_relative_errors),
        "max_fusion_equation_residual": max(fusion_residuals),
        "max_simplex_sum_residual": max(simplex_residuals),
        "nu_one_kappa_one_identity_residual": float(
            np.linalg.norm(identity.final - identity.joint)
        ),
        "equal_norm_covariance_counterexample": {
            "max_norm_difference": float(np.max(np.abs(norm_a - norm_b))),
            "constant_covariance_trace": float(np.trace(covariance_a)),
            "variable_covariance_trace": float(np.trace(covariance_b)),
            "conclusion": (
                "Equal per-update magnitudes can have different gradient-noise "
                "covariance; norm restoration alone does not determine SDE diffusion."
            ),
        },
        "equation10_degeneracy": {
            "mean_norm": float(np.linalg.norm(symmetric_mean)),
            "dual_gradient_norm": float(np.linalg.norm(symmetric_dual)),
            "paper_expression": "rho*||mu|| * g_alpha / ||g_alpha||",
            "is_undefined_without_convention": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--code-root", type=Path, default=CANDIDATE / "official" / "code"
    )
    parser.add_argument(
        "--tex",
        type=Path,
        default=CANDIDATE / "official" / "arxiv-source" / "example_paper.tex",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "evidence" / "claims_audit.json",
    )
    args = parser.parse_args()

    result = {
        "audit_version": 1,
        "source": source_anchors(args.tex),
        "release": release_audit(args.code_root),
        "mechanics": mechanics_audit(),
        "claims": {
            "C1": {
                "verdict": "VERIFIED_EQUATION_LEVEL_WITH_QUALIFICATIONS",
                "prepared_points": 2,
            },
            "C2": {
                "verdict": "NOT_ESTABLISHED_BY_PRINTED_EQUATIONS",
                "prepared_points": 0,
            },
            "C3": {
                "verdict": "INCONCLUSIVE_NOT_RERUN",
                "prepared_points": 0,
            },
            "C4": {
                "verdict": "INCONCLUSIVE_NOT_RERUN",
                "prepared_points": 0,
            },
            "C5": {
                "verdict": "INCONCLUSIVE_NOT_RERUN",
                "prepared_points": 0,
            },
        },
        "prepared_score": 2,
        "score_denominator": 10,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

````


````output
{
  "equal_norm_covariance_counterexample": {
    "conclusion": "Equal per-update magnitudes can have different gradient-noise covariance; norm restoration alone does not determine SDE diffusion.",
    "constant_covariance_trace": 0.0,
    "max_norm_difference": 0.0,
    "variable_covariance_trace": 0.5000000000000001
  },
  "equation10_degeneracy": {
    "dual_gradient_norm": 0.0,
    "is_undefined_without_convention": true,
    "mean_norm": 0.0,
    "paper_expression": "rho*||mu|| * g_alpha / ||g_alpha||"
  },
  "max_fusion_equation_residual": 0.0,
  "max_simplex_sum_residual": 4.973799150320701e-14,
  "max_stage2_magnitude_relative_error": 1.8978682574294423e-12,
  "max_trust_region_boundary_residual": 4.440892098500626e-16,
  "nu_one_kappa_one_identity_residual": 0.0,
  "repeats": 500,
  "seed": 260522635
}
````
