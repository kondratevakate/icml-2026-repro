"""verify_claim2.py — Algorithm 1 / Section 4 of arXiv 2602.01603.

Claim 2: non-linear GRPO replaces the scalar reward by the approximated functional derivative
dR/dpi[pihat_t](y) evaluated at the CURRENT POLICY'S EMPIRICAL distribution, and is otherwise a
drop-in on top of standard GRPO.

Tests
  T1 the analytic derivative (Prop. 4.2, discrete form implemented in iama_core.bon_grad) equals the
     first variation of Definition 4.1, checked by central finite differences along simplex directions:
       d/deps R[pi + eps(pi'-pi)]|_0  vs  <dR/dpi, pi'-pi>.
  T2 "drop-in": running the *standard* GRPO mirror-descent map, unchanged except that the reward
     vector is rtilde = dR/dpi, converges to the maximiser of the true non-linear objective
     (agreement with an independent L-BFGS solver).
  T3 empirical-pihat version (Algorithm 1 line 4: derivative evaluated at the empirical distribution
     of M samples) -> E[||dR/dpi[pihat] - dR/dpi[pi]||_sp^2] decays like O(1/M)  (Section 5.3 lemma).
  MUTATION: keep everything identical but feed the PLAIN reward r (i.e. standard GRPO) instead of the
     functional derivative; the fixed point must then be a strictly worse point of the IAMA objective.

Run: .venv/bin/python verify_claim2.py
"""
import json
import sys
import numpy as np

sys.path.insert(0, ".")
from iama_core import (bon_value, bon_grad, agg_value, agg_grad, exact_step,  # noqa: E402
                       kl, span, solve_optimum)

OUT = "results/claim2.json"
CMD = ".venv/bin/python verify_claim2.py"


def main():
    res = {"command": CMD, "claim": 2, "source": "Algorithm 1 + Proposition 4.2, Section 4, arXiv 2602.01603"}

    # ---------- T1 finite-difference check of the functional derivative ----------
    fd = []
    for K in (4, 6, 10, 20):
        for N in (2, 3, 4, 8):
            for seed in range(5):
                rng = np.random.default_rng(3 * K + 17 * N + seed)
                r = rng.uniform(0, 1, K)
                p = rng.dirichlet(np.full(K, 1.5))
                q = rng.dirichlet(np.full(K, 1.5))
                d = bon_grad(p, r, N)
                analytic = float(np.dot(d, q - p))
                h = 1e-6
                num = (bon_value(p + h * (q - p), r, N) - bon_value(p - h * (q - p), r, N)) / (2 * h)
                fd.append({"K": K, "N": N, "seed": seed, "analytic": analytic, "finite_difference": num,
                           "abs_err": abs(analytic - num),
                           "rel_err": abs(analytic - num) / max(abs(num), 1e-12)})
    res["T1_functional_derivative_vs_finite_difference"] = {
        "n_cases": len(fd), "max_abs_err": float(max(x["abs_err"] for x in fd)),
        "max_rel_err": float(max(x["rel_err"] for x in fd)), "cases": fd[:20],
    }

    # ---------- T2 drop-in convergence + MUTATION ----------
    drop = []
    for K in (6, 10):
        for N in (2, 4, 8):
            for seed in range(5):
                rng = np.random.default_rng(41 * K + 7 * N + seed)
                r0 = rng.uniform(0, 1, K)
                rewards = [r0, 1.0 - r0]
                Ns = [N, N]
                w = np.array([0.5, 0.5])
                beta, eta = 0.05, 1.0
                p_ref = np.full(K, 1.0 / K)

                p = p_ref.copy()
                for _ in range(3000):                      # non-linear GRPO (derivative as reward)
                    p = exact_step(p, agg_grad(p, rewards, Ns, w), beta, eta, p_ref)
                p_star, L_star = solve_optimum(rewards, Ns, w, beta, p_ref, seed=seed)
                L_nl = -agg_value(p, rewards, Ns, w) + beta * kl(p, p_ref)

                d_plain = w[0] * rewards[0] + w[1] * rewards[1]   # MUTATION: standard GRPO reward
                pm = p_ref.copy()
                for _ in range(3000):
                    pm = exact_step(pm, d_plain, beta, eta, p_ref)
                L_mut = -agg_value(pm, rewards, Ns, w) + beta * kl(pm, p_ref)

                drop.append({"K": K, "N": N, "seed": seed,
                             "L_nonlinear_grpo": L_nl, "L_optimum_lbfgs": L_star,
                             "suboptimality_nonlinear_grpo": L_nl - L_star,
                             "tv_to_optimum": float(0.5 * np.abs(p - p_star).sum()),
                             "MUTATION_L_standard_grpo": L_mut,
                             "MUTATION_excess_loss": L_mut - L_star})
    res["T2_dropin_and_mutation"] = {
        "n_instances": len(drop),
        "max_suboptimality_nonlinear_grpo": float(max(d["suboptimality_nonlinear_grpo"] for d in drop)),
        "max_tv_to_optimum": float(max(d["tv_to_optimum"] for d in drop)),
        "min_MUTATION_excess_loss": float(min(d["MUTATION_excess_loss"] for d in drop)),
        "median_MUTATION_excess_loss": float(np.median([d["MUTATION_excess_loss"] for d in drop])),
        "n_instances_mutation_strictly_worse": int(sum(
            1 for d in drop if d["MUTATION_excess_loss"] > d["suboptimality_nonlinear_grpo"] + 1e-9)),
        "instances": drop,
    }

    # ---------- T3 empirical pihat: O(1/M) decay of the span-seminorm error ----------
    emp = {}
    K, N = 20, 4
    rng = np.random.default_rng(2026)
    r = rng.uniform(0, 1, K)
    p = rng.dirichlet(np.full(K, 2.0))
    d_true = bon_grad(p, r, N)
    for M in (8, 16, 32, 64, 128, 256, 512, 1024):
        errs = []
        for rep in range(400):
            rr = np.random.default_rng(10_000 * M + rep)
            idx = rr.choice(K, size=M, p=p)
            phat = np.bincount(idx, minlength=K) / M
            d_hat = bon_grad(phat, r, N)
            errs.append(span(d_hat - d_true) ** 2)
        emp[str(M)] = {"mean_sq_span_error": float(np.mean(errs)),
                       "times_M": float(np.mean(errs) * M)}
    Ms = np.array([8, 16, 32, 64, 128, 256, 512, 1024], float)
    vals = np.array([emp[str(int(m))]["mean_sq_span_error"] for m in Ms])
    slope = float(np.polyfit(np.log(Ms), np.log(vals), 1)[0])
    res["T3_empirical_derivative_error"] = {"per_M": emp, "loglog_slope_vs_M": slope,
                                            "expected_slope": -1.0}

    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({"T1": {k: v for k, v in res["T1_functional_derivative_vs_finite_difference"].items() if k != "cases"},
                      "T2": {k: v for k, v in res["T2_dropin_and_mutation"].items() if k != "instances"},
                      "T3": {"slope": slope, "per_M": emp}}, indent=2))


if __name__ == "__main__":
    main()
