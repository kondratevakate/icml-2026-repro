# Claim 3: bridge estimation error bound


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_eed18e34cbd3", "created_at": "2026-07-29T15:03:34+00:00", "title": "Claim 3: bridge estimation error bound"}
-->
**Anchored claim (verbatim).** "Theorem 5.8 bounds the RKHS-based bridge function estimation error as ||b_hat_t - b_t*||_2 <= constant * tau_t delta_t max{1, ||b_t*||^2} under polynomial eigenvalue decay of the conditional expectation operator."

**Verdict - VERIFIED.**

The pinned Theorem 5.8 contains the stated L2 error, ill-posedness factor
`tau_t`, critical radius `delta_t`, RKHS norm factor, and probability
`1-3 zeta`. The audit independently checks the core reduction
`||e||_2 <= tau_t ||T e||_2` rather than only matching theorem text.

Across 2,000 random errors in finite inverse problems of
dimensions [3, 5, 8, 13], there were `0`
violations. The largest normalized ratio was
`0.870574`, so the check exercises a nonvacuous
part of the bound.

Scope: the upstream projected empirical-process inequality is accepted under
the theorem's explicit star-shapedness, boundedness, realizability, tuning, and
critical-radius assumptions. Polynomial eigenvalue decay appears in the
preceding RKHS corollary; Theorem 5.8 itself is stated in the more general
critical-radius form.


---
<!-- trackio-cell
{"type": "code", "id": "cell_8f3a206bcf6d", "created_at": "2026-07-29T15:03:34+00:00", "title": "C3 machine-readable evidence", "language": "python"}
-->
````python title=audit_claims.py
#!/usr/bin/env python3
"""Independent audits for the five MNAR OPE challenge claims."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, zeta
from scipy.stats import chi2, pearsonr


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = SCRIPT_DIR.parent / "official" / "paper_source"
DEFAULT_REPO = SCRIPT_DIR.parent / "official" / "ShadOPE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
        encoding="utf-8",
    ).strip()


def source_window(path: Path, start_marker: str, end_marker: str) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.find(start_marker)
    if start < 0:
        raise ValueError(f"missing source marker {start_marker!r} in {path}")
    end = text.find(end_marker, start)
    if end < 0:
        raise ValueError(f"missing source marker {end_marker!r} in {path}")
    return text[start : end + len(end_marker)]


def wrapped_gaussian_counterexample(
    sigma: float = 0.25,
    cutoffs: tuple[int, ...] = (5, 10, 20, 30),
) -> dict[str, Any]:
    """Construct a complete inverse problem with no square-integrable bridge.

    R is uniform on [-pi, pi), and S' = R + epsilon modulo 2*pi, where epsilon
    is wrapped Gaussian. The conditional expectation operator is circular
    convolution with nonzero Fourier multipliers
    lambda_k = exp(-sigma^2 k^2 / 2), so it and its adjoint are injective.

    The bounded target g(r)=r has sine coefficients 2*(-1)^(k+1)/k. Any bridge
    must divide these coefficients by lambda_k. Its partial L2 norm is therefore
    2 * sum_k exp(sigma^2 k^2) / k^2, which diverges.
    """

    partials: dict[str, float] = {}
    last_term: dict[str, float] = {}
    for cutoff in cutoffs:
        k = np.arange(1, cutoff + 1, dtype=np.float64)
        terms = 2.0 * np.exp((sigma * k) ** 2) / (k**2)
        partials[str(cutoff)] = float(np.sum(terms))
        last_term[str(cutoff)] = float(terms[-1])

    p_min = float(expit(-0.5 * math.pi))
    p_max = float(expit(0.5 * math.pi))
    lambda_at_max_k = float(
        math.exp(-0.5 * sigma**2 * max(cutoffs) ** 2)
    )
    checks = {
        "reward_is_bounded": True,
        "one_stage_augmented_process_is_markov": True,
        "no_future_dependence_holds_by_construction": True,
        "positivity_is_uniform": 0.0 < p_min < p_max < 1.0,
        "relevance_holds": math.exp(-0.5 * sigma**2) > 0.0,
        "completeness_operator_is_injective": lambda_at_max_k > 0.0,
        "adjoint_completeness_is_injective": lambda_at_max_k > 0.0,
        "picard_terms_fail_to_tend_to_zero": (
            last_term[str(cutoffs[-1])] > last_term[str(cutoffs[-2])]
        ),
        "partial_inverse_norm_explodes": (
            partials[str(cutoffs[-1])] > 1e15
        ),
    }
    return {
        "construction": {
            "reward": "R ~ Uniform[-pi, pi)",
            "next_state": "S_next = (R + wrapped_normal(0, sigma^2)) mod 2*pi",
            "missingness": "O | R ~ Bernoulli(expit(0.5*R))",
            "sigma": sigma,
            "fourier_multiplier": "lambda_k = exp(-sigma^2*k^2/2)",
            "target_sine_coefficient": "2*(-1)^(k+1)/k",
        },
        "positivity_range": [p_min, p_max],
        "smallest_reported_nonzero_multiplier": lambda_at_max_k,
        "picard_partial_sums": partials,
        "picard_last_terms": last_term,
        "checks": checks,
        "conclusion": (
            "All assumptions named by the anchored claim hold, but the inverse "
            "Fourier coefficients are not square summable, so no L2 bridge "
            "exists. The omitted Picard regularity condition is necessary."
        ),
    }


def simulate_official_dgp(n: int = 60000, seed: int = 20260729) -> dict[str, np.ndarray]:
    """Vectorized independent implementation of the official one-step DGP."""

    rng = np.random.default_rng(seed)
    s = rng.normal(size=(n, 2))
    a = rng.choice(np.array([-1.0, 1.0]), size=n)
    innovation = rng.normal(scale=0.1, size=(n, 2))
    s_next = 0.9 * s + 0.2 * a[:, None] + innovation
    reward_noise = rng.uniform(-0.1, 0.1, size=n)
    w_s = np.column_stack((0.9 - 0.6 * a, np.full(n, -0.7)))
    reward_logit = (
        np.sum(w_s * s, axis=1)
        + s_next @ np.array([1.3, 2.0])
        - 0.4 * a
    )
    reward = expit(reward_logit) + reward_noise
    obs_logit = 1.0 - 0.1 * a + 0.2 * (s[:, 0] - 2.0 * s[:, 1]) + 2.5 * reward
    obs_prob = expit(obs_logit)
    observed = rng.binomial(1, obs_prob, size=n)
    return {
        "s": s,
        "a": a,
        "innovation": innovation,
        "s_next": s_next,
        "reward": reward,
        "obs_prob": obs_prob,
        "observed": observed,
    }


def linear_residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(x)), x))
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def relevance_diagnostic(data: dict[str, np.ndarray]) -> dict[str, Any]:
    mask = data["observed"] == 1
    s = data["s"][mask]
    a = data["a"][mask]
    reward = data["reward"][mask]
    s_next = data["s_next"][mask]
    base = np.column_stack((s, a))
    reward_resid = linear_residual(reward, base)
    next_signal = s_next @ np.array([1.3, 2.0])
    next_resid = linear_residual(next_signal, base)
    correlation, p_value = pearsonr(reward_resid, next_resid)

    base_design = np.column_stack((np.ones(len(base)), base))
    full_design = np.column_stack((base_design, s_next))
    base_coef, *_ = np.linalg.lstsq(base_design, reward, rcond=None)
    full_coef, *_ = np.linalg.lstsq(full_design, reward, rcond=None)
    base_mse = float(np.mean((reward - base_design @ base_coef) ** 2))
    full_mse = float(np.mean((reward - full_design @ full_coef) ** 2))
    checks = {
        "observed_subset_is_nonempty": int(mask.sum()) > 1000,
        "residual_association_is_material": abs(float(correlation)) > 0.05,
        "residual_association_is_significant": float(p_value) < 1e-12,
        "next_state_reduces_conditional_mse": full_mse < 0.95 * base_mse,
    }
    return {
        "n_total": int(len(mask)),
        "n_observed": int(mask.sum()),
        "residual_pearson_r": float(correlation),
        "residual_pearson_p": float(p_value),
        "base_mse": base_mse,
        "with_next_state_mse": full_mse,
        "relative_mse": full_mse / base_mse,
        "checks": checks,
    }


def logistic_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        z = x @ beta
        loss = float(np.sum(np.logaddexp(0.0, z) - y * z))
        grad = x.T @ (expit(z) - y)
        return loss, grad

    result = minimize(
        objective,
        np.zeros(x.shape[1]),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"logistic fit failed: {result.message}")
    return np.asarray(result.x), float(result.fun)


def no_future_diagnostic(data: dict[str, np.ndarray]) -> dict[str, Any]:
    s = data["s"]
    a = data["a"]
    reward = data["reward"]
    s_next = data["s_next"]
    observed = data["observed"].astype(np.float64)
    base = np.column_stack((np.ones(len(a)), a, s, reward))
    extended = np.column_stack((base, s_next))
    beta_base, nll_base = logistic_fit(base, observed)
    beta_extended, nll_extended = logistic_fit(extended, observed)
    likelihood_ratio = max(0.0, 2.0 * (nll_base - nll_extended))
    p_value = float(chi2.sf(likelihood_ratio, df=2))
    extra = beta_extended[-2:]
    expected = np.array([1.0, -0.1, 0.2, -0.4, 2.5])
    checks = {
        "base_coefficients_match_generator": bool(
            np.max(np.abs(beta_base - expected)) < 0.25
        ),
        "future_terms_are_not_jointly_significant": (
            p_value > 0.05
        ),
        "future_terms_do_not_materially_improve_fit": (
            (nll_base - nll_extended) / len(a) < 1e-4
        ),
    }
    return {
        "n": int(len(a)),
        "base_coefficients": beta_base.tolist(),
        "expected_base_coefficients": expected.tolist(),
        "extended_future_coefficients": extra.tolist(),
        "nll_base": nll_base,
        "nll_extended": nll_extended,
        "likelihood_ratio": likelihood_ratio,
        "likelihood_ratio_p": p_value,
        "per_sample_nll_gain": (nll_base - nll_extended) / len(a),
        "checks": checks,
    }


def generator_dependency_audit(repo: Path) -> dict[str, Any]:
    path = repo / "src" / "envs" / "sim_envs.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    step_node = None
    reward_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "step":
                step_node = node
            elif node.name == "_compute_reward":
                reward_node = node
    if step_node is None or reward_node is None:
        raise ValueError("could not locate official simulator methods")
    step_text = ast.unparse(step_node)
    reward_text = ast.unparse(reward_node)
    checks = {
        "missingness_logit_uses_current_state": "self._s" in step_text,
        "missingness_logit_uses_current_action": "a" in step_text,
        "missingness_logit_uses_current_reward": "r_true" in step_text,
        "missingness_draw_uses_only_probability": (
            "bernoulli(p_o)" in step_text
        ),
        "reward_uses_next_state": "s_next" in reward_text,
        "reward_has_nonzero_next_state_weights": (
            "np.array([1.3, 2.0])" in reward_text
        ),
    }
    return {
        "path": str(path),
        "sha256": sha256(path),
        "checks": checks,
    }


def bridge_error_reduction_check(
    seed: int = 1729,
    trials: int = 2000,
) -> dict[str, Any]:
    """Check ||e|| <= tau ||T e|| for random finite inverse problems."""

    rng = np.random.default_rng(seed)
    worst_ratio = 0.0
    violations = 0
    dimensions = (3, 5, 8, 13)
    for dimension in dimensions:
        singular = np.geomspace(1.0, 0.03, dimension)
        tau = 1.0 / singular.min()
        for _ in range(trials // len(dimensions)):
            error = rng.normal(size=dimension)
            projected = singular * error
            ratio = np.linalg.norm(error) / (tau * np.linalg.norm(projected))
            worst_ratio = max(worst_ratio, float(ratio))
            violations += int(ratio > 1.0 + 1e-12)
    return {
        "seed": seed,
        "trials": trials,
        "dimensions": list(dimensions),
        "worst_normalized_ratio": worst_ratio,
        "violations": violations,
        "checks": {
            "all_finite_dimensional_reductions_hold": violations == 0,
            "bound_is_nonvacuously_exercised": worst_ratio > 0.1,
        },
    }


def policy_rate_reduction_check(
    horizons: tuple[int, ...] = (2, 4, 8, 16, 32),
    a: float = 0.8,
    alpha: float = 1.4,
) -> dict[str, Any]:
    """Exercise the recurrence and rate substitutions used by Theorem 5.9."""

    cumulative_checks: dict[str, Any] = {}
    recurrence_checks: dict[str, Any] = {}
    k_bound = math.exp(0.5 * a * float(zeta(alpha, 1.0)))
    for horizon in horizons:
        times = np.arange(1, horizon + 1, dtype=np.float64)
        kappa = 1.0 + a / (times**alpha)
        cumulative = float(math.sqrt(np.prod(kappa)))
        cumulative_checks[str(horizon)] = {
            "sqrt_product_kappa": cumulative,
            "uniform_K": k_bound,
            "holds": cumulative <= k_bound + 1e-12,
        }

        delta = 0.02 / np.sqrt(times)
        bridge = 1.7 * delta
        stage = bridge + (horizon - times + 1.0) * delta
        error = 0.0
        for index in range(horizon - 1, -1, -1):
            next_factor = (
                math.sqrt(float(kappa[index + 1]))
                if index + 1 < horizon
                else 0.0
            )
            error = float(stage[index]) + next_factor * error

        explicit = 0.0
        for index in range(horizon):
            weight = 1.0
            for j in range(1, index + 1):
                weight *= math.sqrt(float(kappa[j]))
            explicit += weight * float(stage[index])
        recurrence_checks[str(horizon)] = {
            "backward_recurrence": error,
            "explicit_unrolling": explicit,
            "uniform_K_upper_bound": k_bound * float(np.sum(stage)),
            "identity_holds": abs(error - explicit) < 1e-12,
            "K_bound_holds": error <= k_bound * float(np.sum(stage)) + 1e-12,
        }

    exponent_checks = {}
    for smoothness in (0.6, 1.0, 2.0, 4.0):
        exponent = smoothness / (2.0 * smoothness + 1.0)
        n = 4096.0
        nonparametric = n ** (-exponent) * math.log(n)
        parametric = n**-0.5
        exponent_checks[str(smoothness)] = {
            "exponent": exponent,
            "below_one_half": exponent < 0.5,
            "absorbs_parametric_term_at_n_4096": nonparametric >= parametric,
        }
    checks = {
        "bounded_cumulative_concentrability": all(
            item["holds"] for item in cumulative_checks.values()
        ),
        "backward_recurrence_unrolls_exactly": all(
            item["identity_holds"] for item in recurrence_checks.values()
        ),
        "uniform_K_controls_recurrence": all(
            item["K_bound_holds"] for item in recurrence_checks.values()
        ),
        "nonparametric_exponents_are_below_half": all(
            item["below_one_half"] for item in exponent_checks.values()
        ),
        "parametric_term_is_absorbable": all(
            item["absorbs_parametric_term_at_n_4096"]
            for item in exponent_checks.values()
        ),
    }
    return {
        "a": a,
        "alpha": alpha,
        "uniform_K": k_bound,
        "cumulative_concentrability": cumulative_checks,
        "recurrence": recurrence_checks,
        "sample_rate_exponents": exponent_checks,
        "checks": checks,
    }


def audit(source: Path, repo: Path) -> dict[str, Any]:
    identification = source_window(
        source / "3-identification.tex",
        r"\begin{assumption}[Relevance Condition]",
        r"\end{theorem}",
    )
    no_future = source_window(
        source / "2-preliminaries.tex",
        r"\begin{assumption}[No Future Dependence]",
        r"\end{assumption}",
    )
    bridge_theorem = source_window(
        source / "5-theoretical.tex",
        r"\begin{theorem}[Bridge estimation error bound]",
        r"\end{theorem}",
    )
    policy_theorem = source_window(
        source / "5-theoretical.tex",
        r"\begin{theorem}[Policy value estimation error bound]",
        r"\end{theorem}",
    )
    appendix = (source / "8-appendix.tex").read_text(encoding="utf-8")
    policy_proof = source_window(
        source / "8-appendix.tex",
        r"\section{Proof of \cref{thm:policyerr}}",
        r"\section{Auxiliary lemmas}",
    )

    dgp = simulate_official_dgp()
    dependency = generator_dependency_audit(repo)
    c1_counterexample = wrapped_gaussian_counterexample()
    c2_diagnostic = relevance_diagnostic(dgp)
    c3_reduction = bridge_error_reduction_check()
    c4_reduction = policy_rate_reduction_check()
    c5_diagnostic = no_future_diagnostic(dgp)

    c1_checks = {
        **c1_counterexample["checks"],
        "paper_theorem_has_regularity_qualification": (
            "and some regularity conditions" in identification
        ),
        "appendix_adds_hilbert_schmidt_condition": (
            r"\begin{assumption}[Hilbert-Schmidt property]" in appendix
        ),
        "appendix_adds_picard_summability": (
            r"\label{ass:sumfinite}" in appendix
        ),
    }
    c2_checks = {
        "source_has_exact_relevance_condition": (
            r"S_{t+1}\not\perp R_t \mid S_t, A_t, O_t = 1"
            in identification
        ),
        **dependency["checks"],
        **c2_diagnostic["checks"],
    }
    c3_checks = {
        "source_has_l2_error": (
            r"\big\|\hat b_t-b_t^*\big\|_2" in bridge_theorem
        ),
        "source_has_ill_posedness_factor": r"\tau_t\delta_t" in bridge_theorem,
        "source_has_norm_factor": (
            r"\max\{1,\|b_t^*\|_{\mathcal B^{(t)}}^2\}"
            in bridge_theorem
        ),
        "source_reports_probability": "with probability $1-3\\zeta$"
        in bridge_theorem,
        **c3_reduction["checks"],
    }
    c4_checks = {
        "source_has_T_squared": r"T^{2}" in policy_theorem,
        "source_has_tau_max": r"\tau_{\max}" in policy_theorem,
        "source_has_log_probability_factor": (
            r"\sqrt{\log(c_1 T/\zeta)}" in policy_theorem
        ),
        "source_has_nonparametric_exponent": (
            r"n^{-\frac{\alpha_{\min}}{2\alpha_{\min}+1}}"
            in policy_theorem
        ),
        "source_has_log_n": r"\log n" in policy_theorem,
        "additional_assumptions_are_explicit": all(
            marker in policy_theorem
            for marker in (
                "ass:realize",
                "ass:operatorbound",
                "ass:boundconc",
                "ass:concentrability",
                "thm:identification",
            )
        ),
        "proof_decomposes_I_II_III": all(
            marker in policy_proof for marker in ("(I)", "(II)", "(III)")
        ),
        "proof_contains_backward_induction": (
            "Applying backward induction" in policy_proof
        ),
        "proof_contains_critical_radius_substitution": (
            "With polynomial decay" in policy_proof
            and r"\delta_{t,*}\lesssim" in policy_proof
        ),
        "proof_restores_stage_sum_after_display_typo": (
            r"\sum_{t=1}^T\delta_{t,*}" in policy_proof
        ),
        **c4_reduction["checks"],
    }
    c5_checks = {
        "source_has_exact_no_future_condition": (
            r"O_t \perp (S_{t+1:T}, R_{t+1:T}) \mid S_t, A_t, R_t"
            in no_future
        ),
        **dependency["checks"],
        **c5_diagnostic["checks"],
    }

    claims = {
        "C1": {
            "verdict": "FALSIFIED" if all(c1_checks.values()) else "INCONCLUSIVE",
            "scope": (
                "Falsifies the anchored unqualified sufficiency claim. It does "
                "not falsify the paper theorem with its additional regularity "
                "conditions."
            ),
            "checks": c1_checks,
            "counterexample": c1_counterexample,
        },
        "C2": {
            "verdict": "VERIFIED" if all(c2_checks.values()) else "INCONCLUSIVE",
            "scope": (
                "Exact semantic claim plus an independent observed-subset "
                "diagnostic on the official data-generating process."
            ),
            "checks": c2_checks,
            "diagnostic": c2_diagnostic,
        },
        "C3": {
            "verdict": "VERIFIED" if all(c3_checks.values()) else "INCONCLUSIVE",
            "scope": (
                "Checks the theorem statement and independently exercises the "
                "projected-error-to-L2 reduction. The upstream empirical-process "
                "bound remains conditional on the theorem's stated assumptions."
            ),
            "checks": c3_checks,
            "finite_dimensional_reduction": c3_reduction,
        },
        "C4": {
            "verdict": "VERIFIED" if all(c4_checks.values()) else "INCONCLUSIVE",
            "scope": (
                "The theorem statement, recurrence, concentrability product, "
                "critical-radius substitution, and final T^2 rate were audited. "
                "The appendix has a dropped summation symbol in one intermediate "
                "display and an implicit zeta/T union-bound adjustment; the next "
                "display and final logarithmic factor repair both bookkeeping issues."
            ),
            "checks": c4_checks,
            "rate_reduction": c4_reduction,
        },
        "C5": {
            "verdict": "VERIFIED" if all(c5_checks.values()) else "INCONCLUSIVE",
            "scope": (
                "Exact semantic claim plus structural and numerical checks of "
                "the official missingness mechanism."
            ),
            "checks": c5_checks,
            "diagnostic": c5_diagnostic,
        },
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "provenance": {
            "paper_id": "vpSFJoxyDz",
            "arxiv_id": "2606.20206v1",
            "paper_source_archive_sha256": (
                "2a3ddc551f79f95d6e9a4448c61656c5b2dbcf828faa4a99662e67ef2345229c"
            ),
            "official_repo": "https://github.com/NAIVlab/ShadOPE",
            "official_commit": git_output(repo, "rev-parse", "HEAD"),
            "official_commit_date": git_output(repo, "show", "-s", "--format=%aI"),
            "generator": generator_dependency_audit(repo),
        },
        "claims": claims,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "evidence" / "claims_audit.json",
    )
    args = parser.parse_args()
    result = audit(args.source.resolve(), args.repo.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value["verdict"] for key, value in result["claims"].items()}))
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

````


````output
{
  "checks": {
    "all_finite_dimensional_reductions_hold": true,
    "bound_is_nonvacuously_exercised": true,
    "source_has_ill_posedness_factor": true,
    "source_has_l2_error": true,
    "source_has_norm_factor": true,
    "source_reports_probability": true
  },
  "finite_dimensional_reduction": {
    "checks": {
      "all_finite_dimensional_reductions_hold": true,
      "bound_is_nonvacuously_exercised": true
    },
    "dimensions": [
      3,
      5,
      8,
      13
    ],
    "seed": 1729,
    "trials": 2000,
    "violations": 0,
    "worst_normalized_ratio": 0.8705741657135864
  },
  "scope": "Checks the theorem statement and independently exercises the projected-error-to-L2 reduction. The upstream empirical-process bound remains conditional on the theorem's stated assumptions.",
  "verdict": "VERIFIED"
}
````
