"""verify_claim3.py — Theorem 5.2 (Section 5.1) of arXiv 2602.01603.

Claim: with EXACT proximal updates and step size eta = 1/L,
    L[pi_T] - L[pi_*]  <=  beta * KL[pi_* | pi_0] / ( ((L+beta)/L)^T - 1 ).

Setup: finite response space, weighted-sum-of-BoN reward functional R (Section 4.1 / Prop. 4.2),
L[pi] = -R[pi] + beta KL[pi|pi_ref], pi_0 = pi_ref = uniform.
L is estimated numerically as the smallest constant consistent with Assumption 5.1 over 4000 random
policy pairs, then inflated by a safety factor (L_used = 1.25 * L_hat). Concavity (first inequality
of Assumption 5.1) is checked on the same pairs. pi_* is obtained by an INDEPENDENT solver
(L-BFGS-B on a softmax parametrization), not by the algorithm under test.

Grid (exhaustive): K in {5,8}, N in {2,4,8}, beta in {0.1,0.5,1.0}, 5 reward seeds -> 90 instances,
each checked at every T in 1..25.

MUTATION: rerun with eta = 20/L (violating the theorem's step size). The theorem's guarantee is
only claimed for eta = 1/L; we report whether the bound is then violated.

Run: .venv/bin/python verify_claim3.py
"""
import json
import sys
import numpy as np

sys.path.insert(0, ".")
from iama_core import agg_value, agg_grad, exact_step, kl, estimate_L, solve_optimum, loss  # noqa: E402

OUT = "results/claim3.json"
CMD = ".venv/bin/python verify_claim3.py"
TMAX = 25


def run_instance(K, N, beta, seed, eta_mult=1.0, use_plain_reward=False):
    rng = np.random.default_rng(1009 * K + 101 * N + int(1000 * beta) + seed)
    r0 = rng.uniform(0, 1, K)
    rewards = [r0, 1.0 - r0]
    Ns = [N, N]
    w = np.array([0.5, 0.5])
    p_ref = np.full(K, 1.0 / K)
    L_hat, conc_viol = estimate_L(rewards, Ns, w, p_ref, seed=seed)
    L = 1.25 * max(L_hat, 1e-6)
    eta = eta_mult / L
    p_star, L_opt = solve_optimum(rewards, Ns, w, beta, p_ref, seed=seed)
    kl0 = kl(p_star, p_ref)
    p = p_ref.copy()
    rows = []
    for T in range(1, TMAX + 1):
        if use_plain_reward:                      # MUTATION: standard GRPO reward, not dR/dpi
            d = w[0] * rewards[0] + w[1] * rewards[1]
        else:
            d = agg_grad(p, rewards, Ns, w)
        p = exact_step(p, d, beta, eta, p_ref)
        lhs = loss(p, rewards, Ns, w, beta, p_ref) - L_opt
        rhs = beta * kl0 / (((L + beta) / L) ** T - 1.0)
        rows.append({"T": T, "lhs_excess_loss": float(lhs), "rhs_bound": float(rhs),
                     "ratio_lhs_over_rhs": float(lhs / rhs) if rhs > 0 else float("inf")})
    return {"K": K, "N": N, "beta": beta, "seed": seed, "L_hat": L_hat, "L_used": L,
            "eta": eta, "concavity_max_violation": conc_viol,
            "KL_pistar_pi0": float(kl0), "L_opt": float(L_opt),
            "max_ratio": max(x["ratio_lhs_over_rhs"] for x in rows),
            "final_excess_loss": rows[-1]["lhs_excess_loss"], "rows": rows}


def main():
    res = {"command": CMD, "claim": 3, "source": "Theorem 5.2, Section 5.1, arXiv 2602.01603",
           "TMAX": TMAX, "L_safety_factor": 1.25}
    inst, mut, mut2 = [], [], []
    for K in (5, 8):
        for N in (2, 4, 8):
            for beta in (0.1, 0.5, 1.0):
                for seed in range(5):
                    inst.append(run_instance(K, N, beta, seed, eta_mult=1.0))
                    if seed < 2:
                        m = run_instance(K, N, beta, seed, eta_mult=20.0)
                        mut.append({"K": K, "N": N, "beta": beta, "seed": seed,
                                    "max_ratio": m["max_ratio"],
                                    "final_excess_loss": m["final_excess_loss"]})
                        m2 = run_instance(K, N, beta, seed, eta_mult=1.0, use_plain_reward=True)
                        mut2.append({"K": K, "N": N, "beta": beta, "seed": seed,
                                     "max_ratio": m2["max_ratio"],
                                     "final_excess_loss": m2["final_excess_loss"]})
    res["n_instances"] = len(inst)
    res["max_ratio_over_all_instances_and_T"] = float(max(i["max_ratio"] for i in inst))
    res["n_instances_bound_violated"] = int(sum(1 for i in inst if i["max_ratio"] > 1.0))
    # tolerance-aware: violations whose LHS is above double-precision noise
    TOL = 1e-12
    bad_rows = [r for i in inst for r in i["rows"] if r["ratio_lhs_over_rhs"] > 1.0]
    res["n_rows_total"] = int(sum(len(i["rows"]) for i in inst))
    res["n_rows_bound_violated_raw"] = len(bad_rows)
    res["n_rows_bound_violated_above_1e-12"] = int(sum(1 for r in bad_rows if r["lhs_excess_loss"] > TOL))
    res["max_lhs_among_violating_rows"] = float(max((r["lhs_excess_loss"] for r in bad_rows), default=0.0))
    res["min_T_among_violating_rows"] = int(min((r["T"] for r in bad_rows), default=0))
    res["max_concavity_violation"] = float(max(i["concavity_max_violation"] for i in inst))
    res["max_final_excess_loss"] = float(max(i["final_excess_loss"] for i in inst))
    res["instances"] = inst
    res["MUTATION_eta_20_over_L"] = {
        "n": len(mut), "max_ratio": float(max(m["max_ratio"] for m in mut)),
        "n_violating": int(sum(1 for m in mut if m["max_ratio"] > 1.0)),
        "n_violating_macroscopically": int(sum(1 for m in mut if m["final_excess_loss"] > 1e-3)),
        "max_final_excess_loss": float(max(m["final_excess_loss"] for m in mut)),
        "cases": mut}
    res["MUTATION_plain_reward_standard_grpo"] = {
        "n": len(mut2), "max_ratio": float(max(m["max_ratio"] for m in mut2)),
        "n_violating": int(sum(1 for m in mut2 if m["max_ratio"] > 1.0)),
        "n_violating_macroscopically": int(sum(1 for m in mut2 if m["final_excess_loss"] > 1e-3)),
        "min_final_excess_loss": float(min(m["final_excess_loss"] for m in mut2)),
        "max_final_excess_loss": float(max(m["final_excess_loss"] for m in mut2)),
        "cases": mut2}
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("instances",)}
                     | {"MUTATION_eta_20_over_L":
                        {k: v for k, v in res["MUTATION_eta_20_over_L"].items() if k != "cases"},
                        "MUTATION_plain_reward_standard_grpo":
                        {k: v for k, v in res["MUTATION_plain_reward_standard_grpo"].items() if k != "cases"}},
                     indent=2))


if __name__ == "__main__":
    main()
