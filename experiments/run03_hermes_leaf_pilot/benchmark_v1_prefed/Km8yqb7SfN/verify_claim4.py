"""verify_claim4.py — Theorem 5.3 (Section 5.2) of arXiv 2602.01603.

Claim: with an approximated first-order variation (error eps) and an optimization residual (delta),
inexact non-linear GRPO with eta = 1/L and random index that ~ ((L+beta/2)/L)^t satisfies
    E[ L[pi_that] - L[pi_*] ]  <=  (beta/2) KL[pi_*|pi_0] / ( ((L+beta/2)/L)^T - 1 )  +  2(eps+delta)/beta,
i.e. convergence to a neighbourhood with additive bias 2(eps+delta)/beta.

Implementation
  * derivative noise: rtilde_t = dR/dpi[pi_t] + noise  (i.i.d. uniform, scaled by `noise_scale`);
    eps is MEASURED as the empirical mean of ||noise||_sp^2 over the run, not assumed.
  * optimization residual: after the exact prox solution, the log-policy is perturbed by a random
    vector; the residual r_t = dLhat/dpi[pi_{t+1}] + (1/eta) log(pi_{t+1}/pi_t)
    = -rtilde_t + beta log(pi_{t+1}/pi_ref) + (1/eta) log(pi_{t+1}/pi_t)  is COMPUTED and its
    mean squared span gives delta.
  * the LHS expectation is taken over 200 independent runs and over the theorem's index distribution.
  * pi_* and L_opt come from the independent L-BFGS solver; L from Assumption-5.1 pair sampling (x1.25).

MUTATION: compare against the bound WITHOUT the 2(eps+delta)/beta term (i.e. the exact Theorem 5.2
form). If the additive bias term is necessary, that reduced bound must be violated at large noise.
Also: the measured bias floor must grow as noise grows and shrink with beta (1/beta scaling).

Run: .venv/bin/python verify_claim4.py
"""
import json
import sys
import numpy as np

sys.path.insert(0, ".")
from iama_core import agg_grad, exact_step, kl, span, estimate_L, solve_optimum, loss  # noqa: E402

OUT = "results/claim4.json"
CMD = ".venv/bin/python verify_claim4.py"
T = 20
NRUNS = 200


def one_config(K, N, beta, noise_scale, resid_scale, seed0=0):
    rng0 = np.random.default_rng(31 * K + 7 * N + int(100 * beta))
    r0 = rng0.uniform(0, 1, K)
    rewards, Ns, w = [r0, 1.0 - r0], [N, N], np.array([0.5, 0.5])
    p_ref = np.full(K, 1.0 / K)
    L_hat, _ = estimate_L(rewards, Ns, w, p_ref, seed=0)
    L = 1.25 * max(L_hat, 1e-6)
    eta = 1.0 / L
    p_star, L_opt = solve_optimum(rewards, Ns, w, beta, p_ref, seed=0)
    kl0 = kl(p_star, p_ref)

    wts = np.array([((L + beta / 2) / L) ** t for t in range(1, T + 1)])
    wts /= wts.sum()

    eps_sq, delta_sq, weighted_excess = [], [], []
    for run in range(NRUNS):
        rng = np.random.default_rng(100_000 + 1000 * seed0 + run)
        p = p_ref.copy()
        excess = []
        for t in range(T):
            d = agg_grad(p, rewards, Ns, w)
            noise = rng.uniform(-1, 1, K) * noise_scale
            dhat = d + noise
            eps_sq.append(span(noise) ** 2)
            p_new = exact_step(p, dhat, beta, eta, p_ref)
            if resid_scale > 0:                      # inexact prox solve
                u = rng.uniform(-1, 1, K) * resid_scale
                lg = np.log(p_new) + u
                lg -= lg.max()
                p_new = np.exp(lg); p_new /= p_new.sum()
            r_t = -dhat + beta * np.log(p_new / p_ref) + (1.0 / eta) * np.log(p_new / p)
            delta_sq.append(span(r_t) ** 2)
            p = p_new
            excess.append(loss(p, rewards, Ns, w, beta, p_ref) - L_opt)
        weighted_excess.append(float(np.dot(wts, excess)))

    eps = float(np.mean(eps_sq))
    delta = float(np.mean(delta_sq))
    lhs = float(np.mean(weighted_excess))
    decay = (beta / 2) * kl0 / (((L + beta / 2) / L) ** T - 1.0)
    bias = 2 * (eps + delta) / beta
    return {"K": K, "N": N, "beta": beta, "noise_scale": noise_scale, "resid_scale": resid_scale,
            "L_used": L, "KL_pistar_pi0": float(kl0), "eps_measured": eps, "delta_measured": delta,
            "lhs_E_excess_loss": lhs, "rhs_decay_term": float(decay), "rhs_bias_term": float(bias),
            "rhs_total": float(decay + bias), "ratio_lhs_over_rhs": float(lhs / (decay + bias)),
            "ratio_lhs_over_decay_only_MUTATION": float(lhs / decay) if decay > 0 else float("inf"),
            "sem_lhs": float(np.std(weighted_excess) / np.sqrt(NRUNS))}


def main():
    res = {"command": CMD, "claim": 4, "source": "Theorem 5.3, Section 5.2, arXiv 2602.01603",
           "T": T, "n_runs_per_config": NRUNS, "L_safety_factor": 1.25}
    configs = []
    for K in (5, 8):
        for N in (2, 4):
            for beta in (0.2, 0.5, 1.0):
                for ns, rs in ((0.0, 0.0), (0.05, 0.0), (0.2, 0.0), (0.2, 0.05), (0.5, 0.1)):
                    configs.append(one_config(K, N, beta, ns, rs))
    res["configs"] = configs
    res["n_configs"] = len(configs)
    res["max_ratio_lhs_over_full_bound"] = float(max(c["ratio_lhs_over_rhs"] for c in configs))
    res["n_configs_full_bound_violated"] = int(sum(1 for c in configs if c["ratio_lhs_over_rhs"] > 1.0))
    noisy = [c for c in configs if c["noise_scale"] >= 0.2]
    res["MUTATION_drop_bias_term"] = {
        "n_noisy_configs": len(noisy),
        "n_violating_decay_only_bound": int(sum(1 for c in noisy if c["ratio_lhs_over_decay_only_MUTATION"] > 1.0)),
        "max_ratio_decay_only": float(max(c["ratio_lhs_over_decay_only_MUTATION"] for c in noisy)),
    }
    # bias-floor monotonicity in noise at fixed (K,N,beta)
    mono = []
    for K in (5, 8):
        for N in (2, 4):
            for beta in (0.2, 0.5, 1.0):
                sel = [c for c in configs if (c["K"], c["N"], c["beta"]) == (K, N, beta) and c["resid_scale"] == 0.0]
                sel.sort(key=lambda c: c["noise_scale"])
                vals = [c["lhs_E_excess_loss"] for c in sel]
                mono.append({"K": K, "N": N, "beta": beta, "noise_scales": [c["noise_scale"] for c in sel],
                             "lhs": vals, "monotone_increasing": bool(all(np.diff(vals) > -1e-9))})
    res["bias_floor_monotone_in_noise"] = {
        "n": len(mono), "n_monotone": int(sum(1 for m in mono if m["monotone_increasing"])), "cases": mono}
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k not in ("configs", "bias_floor_monotone_in_noise")}
                     | {"bias_floor_monotone": {k: v for k, v in res["bias_floor_monotone_in_noise"].items() if k != "cases"}},
                     indent=2))


if __name__ == "__main__":
    main()
