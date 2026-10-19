# Claim 5: Kang-Schafer mechanism


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_6546daa0cf8c", "created_at": "2026-07-29T19:11:25+00:00", "title": "Claim 5: Kang-Schafer mechanism"}
-->
**Verdict - PARTIALLY VERIFIED IN A SCOPED INDEPENDENT SIMULATION.**

All `1200` propensity fits converged
in a 30-seed, 10-fold panel. On the paper's nonlinear observed features,
log-loss IPW RMSE is `48.529` versus
`6.833` for the tailored pair, a
`7.10x` ratio; tailored wins
`93.3%` of seeds.

Replacing observed features with the correctly specified latent variables
shrinks the ratio to `1.24x`,
a specification ablation supporting the misspecification mechanism.
Standard-normal
outcome noise is assumed because the paper does not specify its distribution.

Paper anchors: DGP at source lines 794-797; protocol at lines 545-547.


---
<!-- trackio-cell
{"type": "code", "id": "cell_24db8efbf966", "created_at": "2026-07-29T19:11:25+00:00", "title": "C5 machine-readable evidence", "language": "python"}
-->
````python title=audit_claims.py
#!/usr/bin/env python3
"""Independent CPU audits for the tailored causal scoring-rule paper."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp
from scipy.optimize import brentq
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import KFold


SEED = 20260729


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paper_anchors(source: str) -> dict:
    return {
        "C1_bound": "\\label{thm:mse_bound}" in source,
        "C1_fixed_crossfit": (
            "acts as a fixed, deterministic function when evaluating the target unit"
            in source
        ),
        "C1_variance_scaling": (
            "\\mathrm{Var}(\\hat\\tau^{\\mathrm{IPW}}_{\\mathrm{ATE}}) "
            "= \\frac{1}{N} \\mathrm{Var}(Z)"
            in source
        ),
        "C2_curvature": "\\label{eq:curvature_sum}" in source,
        "C3_loss": "\\label{def:ipw_psr}" in source,
        "C4_quartic": "\\label{eq:quartic_eq}" in source,
        "C4_no_vanish": "which cannot explode or vanish" in source,
        "C5_kang": "\\textbf{Kang \\& Schafer.}" in source,
        "C6_acic": "95$ out of $96$ configurations" in source,
    }


def crossfit_scope_audit() -> dict:
    # Two leave-one-out folds. Each held-out contribution depends on the other
    # fold's Bernoulli observation through its fitted prediction.
    rows = []
    for b1, b2 in product((0.0, 1.0), repeat=2):
        q1 = 0.25 + 0.5 * b2
        q2 = 0.25 + 0.5 * b1
        z1 = b1 * q1
        z2 = b2 * q2
        rows.append((z1, z2))
    values = np.asarray(rows)
    covariance = np.cov(values, rowvar=False, ddof=0)
    mean_variance = float(np.var(values.mean(axis=1), ddof=0))
    iid_formula = float(np.var(values[:, 0], ddof=0) / 2)

    # Independently verify the pointwise algebra used by the theorem.
    rng = np.random.default_rng(SEED)
    max_bias_inequality_residual = -np.inf
    max_variance_inequality_residual = -np.inf
    mutation_violations = 0
    for _ in range(5000):
        p = rng.uniform(0.02, 0.98, size=7)
        q = rng.uniform(0.02, 0.98, size=7)
        mu1 = rng.uniform(-2.0, 2.0, size=7)
        mu0 = rng.uniform(-2.0, 2.0, size=7)
        bias = np.mean(mu1 * (p / q - 1) - mu0 * ((1 - p) / (1 - q) - 1))
        k = max(np.mean(mu1**2), np.mean(mu0**2))
        d_bias = np.mean((p / q - 1) ** 2 + ((1 - p) / (1 - q) - 1) ** 2)
        max_bias_inequality_residual = max(
            max_bias_inequality_residual, bias**2 - 2 * k * d_bias
        )

        a = rng.normal(size=200)
        b = rng.normal(size=200)
        variance_residual = np.var(a + b) - 2 * np.var(a) - 2 * np.var(b)
        max_variance_inequality_residual = max(
            max_variance_inequality_residual, variance_residual
        )
        if bias**2 > 0.25 * k * d_bias:
            mutation_violations += 1

    return {
        "exact_crossfit_states": len(rows),
        "crossfit_contribution_covariance": float(covariance[0, 1]),
        "actual_mean_variance": mean_variance,
        "iid_variance_formula": iid_formula,
        "iid_formula_residual": mean_variance - iid_formula,
        "conditional_proof_max_bias_residual": float(
            max_bias_inequality_residual
        ),
        "generic_sum_variance_inequality_max_residual": float(
            max_variance_inequality_residual
        ),
        "weakened_constant_mutation_violations": mutation_violations,
    }


def symbolic_objects() -> dict:
    p, q = sp.symbols("p q", positive=True)
    d_bias = (p / q - 1) ** 2 + ((1 - p) / (1 - q) - 1) ** 2
    d_var = p * (1 / q - 1 / p) ** 2 + (1 - p) * (
        1 / (1 - q) - 1 / (1 - p)
    ) ** 2
    weight = (
        2 / p**2
        + 2 / (1 - p) ** 2
        + 2 / p**3
        + 2 / (1 - p) ** 3
    )
    loss1 = 1 / q**2 - 2 / (1 - q) + 2 * sp.log(q * (1 - q))
    loss0 = 1 / (1 - q) ** 2 - 2 / q + 2 * sp.log(q * (1 - q))
    return {
        "p": p,
        "q": q,
        "d_task": d_bias + d_var,
        "weight": weight,
        "loss1": loss1,
        "loss0": loss0,
    }


def curvature_audit(objects: dict) -> dict:
    p, q = objects["p"], objects["q"]
    observed = sp.simplify(sp.diff(objects["d_task"], q, 2).subs(q, p))
    declared = objects["weight"]
    identity = sp.simplify(observed - declared)
    mutated = declared - 2 / p**3 + 2 / p**2

    grid = np.unique(
        np.concatenate(
            [
                np.geomspace(1e-6, 0.1, 250),
                np.linspace(0.1, 0.9, 500),
                1 - np.geomspace(1e-6, 0.1, 250),
            ]
        )
    )
    mp.mp.dps = 60
    observed_fn = sp.lambdify(p, observed, "mpmath")
    declared_fn = sp.lambdify(p, declared, "mpmath")
    mutated_fn = sp.lambdify(p, mutated, "mpmath")
    relative_errors = []
    mutation_errors = []
    for value in grid:
        point = mp.mpf(str(value))
        observed_value = observed_fn(point)
        declared_value = declared_fn(point)
        scale = max(mp.mpf(1), abs(declared_value))
        relative_errors.append(abs(observed_value - declared_value) / scale)
        mutation_errors.append(abs(observed_value - mutated_fn(point)) / scale)
    return {
        "symbolic_difference": str(identity),
        "grid_size": int(grid.size),
        "max_relative_error": float(max(relative_errors)),
        "mutation_max_relative_error": float(max(mutation_errors)),
    }


def proper_loss_audit(objects: dict) -> dict:
    p, q = objects["p"], objects["q"]
    weight_q = objects["weight"].subs(p, q)
    loss1_residual = sp.simplify(
        sp.diff(objects["loss1"], q) + weight_q * (1 - q)
    )
    loss0_residual = sp.simplify(sp.diff(objects["loss0"], q) - weight_q * q)

    grid_p = np.linspace(0.01, 0.99, 99)
    grid_q = np.linspace(0.001, 0.999, 3993)
    loss1_fn = sp.lambdify(q, objects["loss1"], "numpy")
    loss0_fn = sp.lambdify(q, objects["loss0"], "numpy")
    l1 = loss1_fn(grid_q)
    l0 = loss0_fn(grid_q)
    argmin_errors = []
    mutation_argmin_errors = []
    for truth in grid_p:
        risk = truth * l1 + (1 - truth) * l0
        argmin_errors.append(abs(grid_q[np.argmin(risk)] - truth))
        mutated_risk = truth * l0 + (1 - truth) * l1
        mutation_argmin_errors.append(
            abs(grid_q[np.argmin(mutated_risk)] - truth)
        )
    return {
        "loss1_differential_residual": str(loss1_residual),
        "loss0_differential_residual": str(loss0_residual),
        "minimum_declared_weight_at_half": float(
            objects["weight"].subs(p, sp.Rational(1, 2))
        ),
        "truth_grid_size": int(grid_p.size),
        "prediction_grid_size": int(grid_q.size),
        "max_argmin_abs_error": float(max(argmin_errors)),
        "label_swap_mutation_min_mean_argmin_error": float(
            np.mean(mutation_argmin_errors)
        ),
    }


def canonical_link(p: np.ndarray | float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return (
        -2 / p
        - 1 / p**2
        + 2 / (1 - p)
        + 1 / (1 - p) ** 2
    )


def quartic_probability(z: float) -> tuple[float, float, int]:
    roots = np.roots([1.0, 0.0, -12.0, -16.0, -(z**2)])
    real_roots = sorted(
        float(root.real) for root in roots if abs(root.imag) < 1e-7
    )
    u = real_roots[-1]
    sign = 0.0 if z == 0 else np.sign(z)
    p = 0.5 * (1 + sign * np.sqrt(1 - 4 / u))
    return float(p), u, len(real_roots)


def inverse_link_probability(z: float) -> float:
    if z == 0:
        return 0.5
    if z < 0:
        return float(
            brentq(
                lambda probability: float(canonical_link(probability) - z),
                1e-12,
                0.5,
                xtol=1e-14,
            )
        )
    return float(
        brentq(
            lambda probability: float(canonical_link(probability) - z),
            0.5,
            1 - 1e-12,
            xtol=1e-14,
        )
    )


def canonical_mapping_audit(objects: dict) -> dict:
    p_symbol = objects["p"]
    integrated_derivative = sp.simplify(
        sp.diff(
            -2 / p_symbol
            - 1 / p_symbol**2
            + 2 / (1 - p_symbol)
            + 1 / (1 - p_symbol) ** 2,
            p_symbol,
        )
        - objects["weight"]
    )
    u = sp.symbols("u", positive=True)
    quartic_identity = sp.expand(u * (u + 2) ** 2 * (u - 4)) - (
        u**4 - 12 * u**2 - 16 * u
    )

    logits = np.unique(
        np.concatenate(
            [
                -np.geomspace(1e-6, 1e6, 350),
                np.array([0.0]),
                np.geomspace(1e-6, 1e6, 350),
            ]
        )
    )
    probabilities = []
    residuals = []
    root_counts = []
    root_mapping_errors = []
    for z in logits:
        probability = inverse_link_probability(float(z))
        root_probability, root, count = quartic_probability(float(z))
        probabilities.append(probability)
        residuals.append(
            abs(root**4 - 12 * root**2 - 16 * root - z**2)
            / max(1.0, z**2)
        )
        root_counts.append(count)
        if abs(z) >= 1e-3:
            root_mapping_errors.append(abs(root_probability - probability))
    probabilities = np.asarray(probabilities)
    roundtrip = canonical_link(probabilities)
    scaled_residual = np.abs(roundtrip - logits) / np.maximum(1.0, np.abs(logits))

    gradient_y1 = probabilities - 1.0
    mutation_logits = logits[np.abs(logits) <= 20]
    sigmoid = 1 / (1 + np.exp(-mutation_logits))
    weight_fn = sp.lambdify(p_symbol, objects["weight"], "numpy")
    mismatched_gradient = (
        weight_fn(np.clip(sigmoid, 1e-12, 1 - 1e-12))
        * (sigmoid - (mutation_logits >= 0).astype(float))
        * sigmoid
        * (1 - sigmoid)
    )
    return {
        "integrated_link_derivative_residual": str(integrated_derivative),
        "quartic_symbolic_residual": str(sp.expand(quartic_identity)),
        "logit_grid_size": int(logits.size),
        "max_roundtrip_relative_error": float(np.max(scaled_residual)),
        "strictly_monotone": bool(np.all(np.diff(probabilities) > 0)),
        "max_symmetry_error": float(
            np.max(np.abs(probabilities + probabilities[::-1] - 1))
        ),
        "minimum_largest_root": float(
            min(quartic_probability(float(z))[1] for z in logits)
        ),
        "real_root_count_range": [min(root_counts), max(root_counts)],
        "quartic_max_relative_residual": float(max(residuals)),
        "quartic_root_vs_inverse_link_max_abs_error": float(
            max(root_mapping_errors)
        ),
        "canonical_gradient_abs_at_z_1e6_y1": float(abs(gradient_y1[-1])),
        "mismatched_sigmoid_max_abs_gradient": float(
            np.nanmax(np.abs(mismatched_gradient))
        ),
    }


def inverse_link_vector(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    lower = np.full(logits.shape, 1e-8)
    upper = np.full(logits.shape, 1 - 1e-8)
    for _ in range(48):
        midpoint = (lower + upper) / 2
        move_lower = canonical_link(midpoint) < logits
        lower = np.where(move_lower, midpoint, lower)
        upper = np.where(move_lower, upper, midpoint)
    return (lower + upper) / 2


def tailored_loss(probability: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-8, 1 - 1e-8)
    common = 2 * np.log(probability * (1 - probability))
    loss1 = 1 / probability**2 - 2 / (1 - probability) + common
    loss0 = 1 / (1 - probability) ** 2 - 2 / probability + common
    return treatment * loss1 + (1 - treatment) * loss0


def fit_propensity(
    design: np.ndarray, treatment: np.ndarray, kind: str, regularization: float
) -> tuple[np.ndarray, bool]:
    n_samples, n_features = design.shape

    def objective(coef: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ coef
        if kind == "log":
            probability = expit(logits)
            value = np.mean(np.logaddexp(0, logits) - treatment * logits)
        else:
            probability = inverse_link_vector(logits)
            value = np.mean(tailored_loss(probability, treatment))
        value += 0.5 * regularization * np.dot(coef[1:], coef[1:])
        gradient = design.T @ (probability - treatment) / n_samples
        gradient[1:] += regularization * coef[1:]
        return float(value), gradient

    result = minimize(
        objective,
        np.zeros(n_features),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 250, "ftol": 1e-10, "gtol": 1e-7},
    )
    return result.x, bool(result.success)


def crossfit_propensity(
    features: np.ndarray,
    treatment: np.ndarray,
    kind: str,
    seed: int,
    folds: int = 10,
    regularization: float = 1e-3,
) -> tuple[np.ndarray, int]:
    prediction = np.empty(treatment.size)
    successful_fits = 0
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, test in splitter.split(features):
        mean = features[train].mean(axis=0)
        scale = features[train].std(axis=0)
        scale[scale < 1e-12] = 1.0
        train_design = np.column_stack(
            [np.ones(train.size), (features[train] - mean) / scale]
        )
        test_design = np.column_stack(
            [np.ones(test.size), (features[test] - mean) / scale]
        )
        coef, success = fit_propensity(
            train_design, treatment[train], kind, regularization
        )
        successful_fits += int(success)
        logits = test_design @ coef
        prediction[test] = (
            expit(logits) if kind == "log" else inverse_link_vector(logits)
        )
    return prediction, successful_fits


def kang_schafer_audit() -> dict:
    seeds = np.arange(SEED, SEED + 30, dtype=np.int64)
    errors = {
        "observed_log": [],
        "observed_tailored": [],
        "latent_log": [],
        "latent_tailored": [],
    }
    boundary_rates = {key: [] for key in errors}
    successful_fits = {key: 0 for key in errors}

    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        n = 2000
        latent = rng.normal(size=(n, 4))
        true_propensity = expit(
            -latent[:, 0]
            + 0.5 * latent[:, 1]
            - 0.25 * latent[:, 2]
            - 0.1 * latent[:, 3]
        )
        treatment = rng.binomial(1, true_propensity)
        baseline = (
            210
            + 27.4 * latent[:, 0]
            + 13.7 * latent[:, 1]
            + 13.7 * latent[:, 2]
            + 13.7 * latent[:, 3]
            + rng.normal(size=n)
        )
        outcome = baseline + 10 * treatment
        observed = np.column_stack(
            [
                np.exp(latent[:, 0] / 2),
                latent[:, 1] / (1 + np.exp(latent[:, 0])) + 10,
                (latent[:, 0] * latent[:, 2] / 25 + 0.6) ** 3,
                (latent[:, 1] + latent[:, 3] + 20) ** 2,
            ]
        )

        for feature_name, features in (("observed", observed), ("latent", latent)):
            for kind_name, kind in (("log", "log"), ("tailored", "tailored")):
                key = f"{feature_name}_{kind_name}"
                propensity, successes = crossfit_propensity(
                    features, treatment, kind, int(seed)
                )
                successful_fits[key] += successes
                boundary_rates[key].append(
                    float(np.mean((propensity < 0.01) | (propensity > 0.99)))
                )
                stable = np.clip(propensity, 1e-6, 1 - 1e-6)
                estimate = np.mean(
                    treatment * outcome / stable
                    - (1 - treatment) * outcome / (1 - stable)
                )
                errors[key].append(float(estimate - 10))

    def summarize(values: list[float]) -> dict:
        array = np.asarray(values)
        return {
            "bias": float(array.mean()),
            "mae": float(np.mean(np.abs(array))),
            "rmse": float(np.sqrt(np.mean(array**2))),
            "max_abs_error": float(np.max(np.abs(array))),
        }

    summaries = {key: summarize(values) for key, values in errors.items()}
    observed_ratio = (
        summaries["observed_log"]["rmse"]
        / summaries["observed_tailored"]["rmse"]
    )
    latent_ratio = (
        summaries["latent_log"]["rmse"] / summaries["latent_tailored"]["rmse"]
    )
    return {
        "seed_count": int(seeds.size),
        "seed_sha256": hashlib.sha256(seeds.tobytes()).hexdigest(),
        "samples_per_seed": 2000,
        "crossfit_folds": 10,
        "regularization": 1e-3,
        "assumed_noise": "standard_normal_not_specified_in_paper",
        "metrics": summaries,
        "mean_boundary_rates": {
            key: float(np.mean(values)) for key, values in boundary_rates.items()
        },
        "successful_fit_count": successful_fits,
        "expected_fit_count_per_method": int(seeds.size * 10),
        "observed_rmse_ratio_log_over_tailored": float(observed_ratio),
        "latent_rmse_ratio_log_over_tailored": float(latent_ratio),
        "misspecification_amplification": float(observed_ratio / latent_ratio),
        "observed_tailored_abs_error_win_rate": float(
            np.mean(
                np.abs(errors["observed_tailored"])
                < np.abs(errors["observed_log"])
            )
        ),
    }


def run_audit(source_path: Path) -> dict:
    source = source_path.read_text(encoding="utf-8", errors="replace")
    anchors = paper_anchors(source)
    objects = symbolic_objects()
    c1 = crossfit_scope_audit()
    c2 = curvature_audit(objects)
    c3 = proper_loss_audit(objects)
    c4 = canonical_mapping_audit(objects)
    c5 = kang_schafer_audit()
    c1["paper_anchors_present"] = all(
        anchors[key] for key in ("C1_bound", "C1_fixed_crossfit", "C1_variance_scaling")
    )
    c2["paper_anchor_present"] = anchors["C2_curvature"]
    c3["paper_anchor_present"] = anchors["C3_loss"]
    c4["paper_anchors_present"] = anchors["C4_quartic"] and anchors["C4_no_vanish"]
    c5["paper_anchor_present"] = anchors["C5_kang"]

    claims = [
        {
            "id": "C1",
            "verdict": "PARTIALLY_VERIFIED_CONDITIONAL_ON_FIXED_PROPENSITY",
            "score": 1,
            "evidence": c1,
        },
        {
            "id": "C2",
            "verdict": "VERIFIED",
            "score": 2,
            "evidence": c2,
        },
        {
            "id": "C3",
            "verdict": "VERIFIED",
            "score": 2,
            "evidence": c3,
        },
        {
            "id": "C4",
            "verdict": "PARTIALLY_VERIFIED_NO_VANISH_CLAUSE_FALSIFIED",
            "score": 1,
            "evidence": c4,
        },
        {
            "id": "C5",
            "verdict": "PARTIALLY_VERIFIED_SCOPED_INDEPENDENT_SIMULATION",
            "score": 1,
            "evidence": c5,
        },
        {
            "id": "C6",
            "verdict": "INCONCLUSIVE_NOT_EXECUTED",
            "score": 0,
            "evidence": {
                "paper_anchor_present": anchors["C6_acic"],
                "author_code_found": False,
            },
        },
    ]
    return {
        "paper": {
            "title": "Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference",
            "forum_id": "JTwryHNicJ",
            "arxiv": "2606.03332v1",
        },
        "provenance": {
            "paper_source_sha256": sha256(source_path),
            "source_archive_sha256": (
                "7862c54d7da9cabecfefbc7bd2308c7cdff507223be68ed9c9ebaada73fe1129"
            ),
        },
        "audit_seed": SEED,
        "claims": claims,
        "prepared_score": sum(claim["score"] for claim in claims),
        "maximum_score": 12,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(args.paper_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Prepared {result['prepared_score']}/{result['maximum_score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

````


````output
{
  "evidence": {
    "assumed_noise": "standard_normal_not_specified_in_paper",
    "crossfit_folds": 10,
    "expected_fit_count_per_method": 300,
    "latent_rmse_ratio_log_over_tailored": 1.2369334134354812,
    "mean_boundary_rates": {
      "latent_log": 0.00016666666666666666,
      "latent_tailored": 0.0,
      "observed_log": 0.004600000000000001,
      "observed_tailored": 0.0
    },
    "metrics": {
      "latent_log": {
        "bias": 0.13409323270299406,
        "mae": 4.470379079942446,
        "max_abs_error": 11.625395877055144,
        "rmse": 5.4395135689767296
      },
      "latent_tailored": {
        "bias": -3.880668553198649,
        "mae": 3.9981085484637147,
        "max_abs_error": 7.423993447234377,
        "rmse": 4.397579942374526
      },
      "observed_log": {
        "bias": 34.70529332937124,
        "mae": 34.70529332937124,
        "max_abs_error": 153.416019010626,
        "rmse": 48.52937765742344
      },
      "observed_tailored": {
        "bias": -6.55449102469592,
        "mae": 6.55449102469592,
        "max_abs_error": 9.975654487802501,
        "rmse": 6.8325366238235565
      }
    },
    "misspecification_amplification": 5.7421751097359195,
    "observed_rmse_ratio_log_over_tailored": 7.10268825902991,
    "observed_tailored_abs_error_win_rate": 0.9333333333333333,
    "paper_anchor_present": true,
    "regularization": 0.001,
    "samples_per_seed": 2000,
    "seed_count": 30,
    "seed_sha256": "b9727908367ea1475e75f3122df53b9b984a131bc7a2817e6beec661ec20c37b",
    "successful_fit_count": {
      "latent_log": 300,
      "latent_tailored": 300,
      "observed_log": 300,
      "observed_tailored": 300
    }
  },
  "id": "C5",
  "score": 1,
  "verdict": "PARTIALLY_VERIFIED_SCOPED_INDEPENDENT_SIMULATION"
}
````
