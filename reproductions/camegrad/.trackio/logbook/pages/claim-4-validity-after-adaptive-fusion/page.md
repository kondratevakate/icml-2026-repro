# Claim 4: validity after adaptive fusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_692dfadbb4f8", "created_at": "2026-07-29T18:36:44+00:00", "title": "Claim 4: validity after adaptive fusion"}
-->
**Verdict - QUALIFIED SCOPE LIMITATION AFTER ADAPTIVE FUSION.**

The fixed two-task construction solves the declared Stage 1 dual at boundary
`alpha=0.0`; its right derivative is positive
(`0.326064`), confirming the boundary
minimum. Stage 1 inner products are
`[2.6645898033750313, 2.338525491562421]`, both positive. Stage 2 reaches its target
norm to numerical tolerance. With legal weights `[0.8, 0.2]`
and `nu=0.9`, Stage 3 produces final inner products
`[5.738862501316584, -0.24260390681533428]` and reintroduces conflict with task 2.

This does not falsify the paper's formal glossary definition, which applies
geometric validity to the rectified Stage 1 direction. It qualifies the broad
algorithm-level narrative; the appendix itself acknowledges conflict
resurgence for large fusion coefficients.


---
<!-- trackio-cell
{"type": "code", "id": "cell_f0ea9415cd65", "created_at": "2026-07-29T18:36:44+00:00", "title": "C4 machine-readable evidence", "language": "python"}
-->
````python title=audit_claims.py
#!/usr/bin/env python3
"""Independent CPU audits of CAME-Grad's mathematical claims."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


COMMIT = "79059e39060d13ef6b6cb2ea9ad7a5d8e2519c83"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interaction_audit(rng: np.random.Generator) -> dict:
    max_error = 0.0
    max_sign_mutation_error = 0.0
    for _ in range(1000):
        g0 = rng.normal(size=17)
        g1 = rng.normal(size=17)
        w0, w1 = rng.uniform(0.05, 2.0, size=2)
        observed = np.linalg.norm(w0 * g0 + w1 * g1) ** 2
        declared = (
            w0**2 * np.dot(g0, g0)
            + w1**2 * np.dot(g1, g1)
            + 2 * w0 * w1 * np.dot(g0, g1)
        )
        mutated = (
            w0**2 * np.dot(g0, g0)
            + w1**2 * np.dot(g1, g1)
            - 2 * w0 * w1 * np.dot(g0, g1)
        )
        max_error = max(max_error, abs(observed - declared))
        max_sign_mutation_error = max(
            max_sign_mutation_error, abs(observed - mutated)
        )

    opposing = np.array([1.0, -2.0, 3.0])
    cancelled = opposing + (-opposing)
    return {
        "random_trials": 1000,
        "identity_max_abs_error": max_error,
        "sign_mutation_max_abs_error": max_sign_mutation_error,
        "opposing_individual_energy": float(
            np.dot(opposing, opposing) + np.dot(-opposing, -opposing)
        ),
        "opposing_joint_energy": float(np.dot(cancelled, cancelled)),
    }


def stage1_counterexample() -> dict:
    # g1=1 and g2=-10 give mu=-4.5. With rho=0.5 the trust region is
    # [-6.75, -2.25]. The max-min solution is its right endpoint -2.25.
    gradients = np.array([1.0, -10.0])
    mu = float(gradients.mean())
    rho = 0.5
    radius = rho * abs(mu)
    lower, upper = mu - radius, mu + radius
    u_star = upper
    improvements = gradients * u_star

    alpha_star = np.array([1.0, 0.0])
    g_alpha = float(alpha_star @ gradients)
    dual_objective = g_alpha * mu + radius * abs(g_alpha)
    recovered = mu + radius * np.sign(g_alpha)
    return {
        "gradients": gradients.tolist(),
        "mu": mu,
        "rho": rho,
        "trust_region": [lower, upper],
        "primal_u_star": u_star,
        "task_inner_products": improvements.tolist(),
        "worst_inner_product": float(improvements.min()),
        "dual_alpha_star": alpha_star.tolist(),
        "dual_objective": dual_objective,
        "dual_recovered_u": recovered,
        "trust_region_residual": abs(u_star - mu) - radius,
    }


def stage2_audit(rng: np.random.Generator) -> dict:
    u = np.array([3.0e-8, 4.0e-8])
    joint = np.array([3.0, 4.0])
    kappa = 1.5
    epsilon = 1.0e-8
    target = kappa * np.linalg.norm(joint)
    enhanced = u * target / (np.linalg.norm(u) + epsilon)

    samples = rng.normal(size=(200000, 5))
    trace_before = float(np.trace(np.cov(samples, rowvar=False, ddof=1)))
    trace_after = float(
        np.trace(np.cov(kappa * samples, rowvar=False, ddof=1))
    )
    return {
        "kappa": kappa,
        "epsilon": epsilon,
        "target_norm": target,
        "observed_norm": float(np.linalg.norm(enhanced)),
        "relative_target_shortfall": float(
            (target - np.linalg.norm(enhanced)) / target
        ),
        "covariance_trace_before": trace_before,
        "covariance_trace_after": trace_after,
        "observed_trace_ratio": trace_after / trace_before,
        "declared_trace_ratio": kappa**2,
    }


def stage4_counterexample() -> dict:
    # This fixed example executes all three declared stages. For the Stage 1
    # convex dual, the right derivative at alpha=0 is positive, so alpha=0 is
    # the exact boundary minimizer.
    gradients = np.array([[-2.0, -2.0], [-1.0, 2.0]])
    mu = gradients.mean(axis=0)
    rho = 0.25
    radius = rho * np.linalg.norm(mu)
    alpha_star = 0.0
    g_alpha = alpha_star * gradients[0] + (1 - alpha_star) * gradients[1]
    dual_right_derivative = float(
        np.dot(gradients[0] - gradients[1], mu)
        + radius
        * np.dot(g_alpha, gradients[0] - gradients[1])
        / np.linalg.norm(g_alpha)
    )
    u_rect = mu + radius * g_alpha / np.linalg.norm(g_alpha)
    stage1_products = gradients @ u_rect

    weights = np.array([0.8, 0.2])
    g_joint = weights @ gradients
    kappa = 1.0
    epsilon = 1.0e-12
    target_norm = kappa * np.linalg.norm(g_joint)
    u_en = u_rect * target_norm / (np.linalg.norm(u_rect) + epsilon)
    nu = 0.9
    final = (1 - nu) * u_en + nu * (kappa * g_joint)
    final_products = gradients @ final
    return {
        "gradients": gradients.tolist(),
        "mu": mu.tolist(),
        "rho": rho,
        "trust_region_radius": float(radius),
        "dual_alpha_star": alpha_star,
        "dual_right_derivative_at_zero": dual_right_derivative,
        "stage1_direction": u_rect.tolist(),
        "stage1_inner_products": stage1_products.tolist(),
        "task_weights": weights.tolist(),
        "joint_gradient": g_joint.tolist(),
        "kappa": kappa,
        "epsilon": epsilon,
        "stage2_target_norm": float(target_norm),
        "stage2_direction": u_en.tolist(),
        "stage2_observed_norm": float(np.linalg.norm(u_en)),
        "nu": nu,
        "final_direction": final.tolist(),
        "final_inner_products": final_products.tolist(),
        "stage1_is_geometrically_valid": bool(np.all(stage1_products >= 0)),
        "final_is_geometrically_valid": bool(np.all(final_products >= 0)),
    }


def run_audit(code_root: Path, source_path: Path) -> dict:
    rng = np.random.default_rng(20260729)
    trainer = code_root / "modules" / "trainer.py"
    optimizer = code_root / "modules" / "CAME_Grad.py"
    readme = code_root / "README.md"
    source = source_path.read_text(encoding="utf-8", errors="replace")
    paper_anchors = {
        "C1": "\\|\\mathbf{g}_{joint}\\|^2 \\approx" in source,
        "C2": (
            "forces the generation of a consensus direction possessing a "
            "positive inner product with all task gradients"
            in source
        ),
        "C3": "By strictly enforcing the target magnitude" in source,
        "C4_formula": (
            "\\mathbf{g}_{final} = (1 - \\nu) \\mathbf{u}_{en} "
            "+ \\nu \\mathbf{g}'_{joint}"
            in source
        ),
        "C4_rectified_scope": (
            "inner product between the rectified update vector "
            "$\\mathbf{u}_{rect}^*$ and each task gradient"
            in source
        ),
        "C4_known_resurgence": (
            "increasing $\\nu$ beyond 0.3 for these models leads to a "
            "resurgence of conflict issues"
            in source
        ),
    }

    c1 = interaction_audit(rng)
    c2 = stage1_counterexample()
    c3 = stage2_audit(rng)
    c4 = stage4_counterexample()
    c1["paper_anchor_present"] = paper_anchors["C1"]
    c2["paper_anchor_present"] = paper_anchors["C2"]
    c3["paper_anchor_present"] = paper_anchors["C3"]
    c4["paper_formula_anchor_present"] = paper_anchors["C4_formula"]
    c4["paper_defines_validity_for_rectified_direction"] = paper_anchors[
        "C4_rectified_scope"
    ]
    c4["paper_acknowledges_conflict_resurgence"] = paper_anchors[
        "C4_known_resurgence"
    ]
    claims = [
        {
            "id": "C1",
            "verdict": "VERIFIED_FOR_EXACT_TWO_TASK_IDENTITY",
            "score": 2,
            "evidence": c1,
        },
        {
            "id": "C2",
            "verdict": "FALSIFIED_AS_UNCONDITIONAL_GUARANTEE",
            "score": 2,
            "evidence": c2,
        },
        {
            "id": "C3",
            "verdict": "PARTIALLY_VERIFIED_WITH_EPSILON_QUALIFICATION",
            "score": 1,
            "evidence": c3,
        },
        {
            "id": "C4",
            "verdict": "QUALIFIED_SCOPE_LIMITATION_AFTER_ADAPTIVE_FUSION",
            "score": 1,
            "evidence": c4,
        },
        {
            "id": "C5",
            "verdict": "INCONCLUSIVE_NOT_EXECUTED",
            "score": 0,
            "evidence": {
                "author_optimizer_file_present": optimizer.exists(),
                "trainer_imports_missing_optimizer": (
                    "from .CAME_Grad import CAME_Grad"
                    in trainer.read_text(encoding="utf-8")
                ),
                "readme_says_core_withheld": (
                    "core optimizer source code is temporarily withheld"
                    in readme.read_text(encoding="utf-8")
                ),
                "paper_value_anchors_present": (
                    "2.3\\%" in source and "1.9\\%" in source
                ),
            },
        },
    ]
    return {
        "paper": {
            "title": "The Double Dilemma in Multi-Task Radiology Report Generation: A Gradient Dynamics Analysis and Solution",
            "forum_id": "T7y2wavrFM",
            "arxiv": "2605.22635v2",
        },
        "provenance": {
            "official_commit": COMMIT,
            "trainer_sha256": sha256(trainer),
            "readme_sha256": sha256(readme),
            "paper_source_sha256": sha256(source_path),
        },
        "audit_seed": 20260729,
        "claims": claims,
        "prepared_score": sum(claim["score"] for claim in claims),
        "maximum_score": 10,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-code-root", type=Path, required=True)
    parser.add_argument("--paper-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(args.official_code_root, args.paper_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Prepared {result['prepared_score']}/{result['maximum_score']}")


if __name__ == "__main__":
    main()

````


````output
{
  "evidence": {
    "dual_alpha_star": 0.0,
    "dual_right_derivative_at_zero": 0.32606431181261053,
    "epsilon": 1e-12,
    "final_direction": [
      -1.832086198167083,
      -1.0373450524912087
    ],
    "final_inner_products": [
      5.738862501316584,
      -0.24260390681533428
    ],
    "final_is_geometrically_valid": false,
    "gradients": [
      [
        -2.0,
        -2.0
      ],
      [
        -1.0,
        2.0
      ]
    ],
    "joint_gradient": [
      -1.8,
      -1.2000000000000002
    ],
    "kappa": 1.0,
    "mu": [
      -1.5,
      0.0
    ],
    "nu": 0.9,
    "paper_acknowledges_conflict_resurgence": true,
    "paper_defines_validity_for_rectified_direction": true,
    "paper_formula_anchor_present": true,
    "rho": 0.25,
    "stage1_direction": [
      -1.6677050983124841,
      0.33541019662496846
    ],
    "stage1_inner_products": [
      2.6645898033750313,
      2.338525491562421
    ],
    "stage1_is_geometrically_valid": true,
    "stage2_direction": [
      -2.12086198167083,
      0.42654947508791713
    ],
    "stage2_observed_norm": 2.163330765277122,
    "stage2_target_norm": 2.1633307652783937,
    "task_weights": [
      0.8,
      0.2
    ],
    "trust_region_radius": 0.375
  },
  "id": "C4",
  "score": 1,
  "verdict": "QUALIFIED_SCOPE_LIMITATION_AFTER_ADAPTIVE_FUSION"
}
````
