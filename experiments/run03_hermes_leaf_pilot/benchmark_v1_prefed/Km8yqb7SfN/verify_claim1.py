"""verify_claim1.py — Equation (1) / Section 3 of arXiv 2602.01603.

Claim 1 (three testable sub-statements):
  (a) The IAMA objective R[pi] = g(R_1[pi],...,R_m[pi]) - beta KL[pi|pi_ref] can be instantiated with
      R_i[pi] = E_{y ~ T_i[pi]}[r_i(y)] for an inference-time transform T_i (here BoN_N).
  (b) R[pi] is NON-LINEAR in pi because of T_i, even for weighted-sum g — in contrast to E_pi[r]
      which is exactly linear. Measured as the chord-vs-curve gap
      D(t) = R[(1-t)pa + t pb] - [(1-t)R[pa] + t R[pb]], maximised over t.
  (c) A single base policy trained on the aggregated objective adapts at inference time to several
      criteria: for each i, R_i[trained base under T_i] beats R_i of the naive (transform-unaware)
      base under T_i. Exhaustive over an instance grid (K, N, m, seeds).
  MUTATION: N=1 makes BoN the identity transform, so R must become exactly linear (gap ~ 0) and the
  IAMA optimum must coincide with the naive optimum.

Run: .venv/bin/python verify_claim1.py
"""
import json
import sys
import numpy as np

sys.path.insert(0, ".")
from iama_core import (bon_value, linear_value, agg_value, agg_grad, exact_step,  # noqa: E402
                       kl, solve_optimum)

OUT = "results/claim1.json"
CMD = ".venv/bin/python verify_claim1.py"


def train(rewards, Ns, w, beta, p_ref, iters=3000, eta=1.0):
    p = p_ref.copy()
    for _ in range(iters):
        d = agg_grad(p, rewards, Ns, w)
        p = exact_step(p, d, beta, eta, p_ref)
    return p


def train_naive(rewards, w, beta, p_ref, iters=3000, eta=1.0):
    """Standard alignment: maximise sum_i w_i E_pi[r_i] - beta KL (linear reward, no transform)."""
    p = p_ref.copy()
    d = sum(wi * ri for wi, ri in zip(w, rewards))
    for _ in range(iters):
        p = exact_step(p, d, beta, eta, p_ref)
    return p


def main():
    rng_master = np.random.default_rng(20260801)
    res = {"command": CMD, "claim": 1, "source": "Equation (1), Section 3, arXiv 2602.01603"}

    # ---------- (b) non-linearity of R in pi ----------
    nonlin = []
    lin_check = []
    mutation_N1 = []
    ts = np.linspace(0.05, 0.95, 19)
    for K in (4, 6, 10):
        for N in (2, 4, 8):
            for seed in range(5):
                rng = np.random.default_rng(1000 * K + 10 * N + seed)
                r = rng.uniform(0, 1, K)
                pa = rng.dirichlet(np.full(K, 1.0))
                pb = rng.dirichlet(np.full(K, 1.0))
                Ra, Rb = bon_value(pa, r, N), bon_value(pb, r, N)
                gaps = [abs(bon_value((1 - t) * pa + t * pb, r, N) - ((1 - t) * Ra + t * Rb)) for t in ts]
                nonlin.append({"K": K, "N": N, "seed": seed, "max_chord_gap": float(max(gaps))})
                La, Lb = linear_value(pa, r), linear_value(pb, r)
                lgaps = [abs(linear_value((1 - t) * pa + t * pb, r) - ((1 - t) * La + t * Lb)) for t in ts]
                lin_check.append(float(max(lgaps)))
                # MUTATION: N = 1
                Ra1, Rb1 = bon_value(pa, r, 1), bon_value(pb, r, 1)
                g1 = [abs(bon_value((1 - t) * pa + t * pb, r, 1) - ((1 - t) * Ra1 + t * Rb1)) for t in ts]
                mutation_N1.append(float(max(g1)))
    res["b_nonlinearity"] = {
        "instances": nonlin,
        "min_max_chord_gap_over_instances": float(min(d["max_chord_gap"] for d in nonlin)),
        "max_max_chord_gap_over_instances": float(max(d["max_chord_gap"] for d in nonlin)),
        "linear_reward_max_chord_gap": float(max(lin_check)),
        "MUTATION_N1_max_chord_gap": float(max(mutation_N1)),
    }

    # ---------- (c) one base policy adapts to m criteria ----------
    adapt = []
    for K in (6, 10):
        for N in (2, 4, 8):
            for m in (2, 3):
                for seed in range(5):
                    rng = np.random.default_rng(7 * K + 13 * N + 31 * m + seed)
                    rewards = [rng.uniform(0, 1, K) for _ in range(m)]
                    # make criteria conflicting: r_2 anti-correlated with r_1
                    rewards[1] = 1.0 - rewards[0]
                    Ns = [N] * m
                    w = np.full(m, 1.0 / m)
                    beta = 0.05
                    p_ref = np.full(K, 1.0 / K)
                    p_iama = train(rewards, Ns, w, beta, p_ref)
                    p_naive = train_naive(rewards, w, beta, p_ref)
                    per_i = []
                    for i in range(m):
                        per_i.append({"i": i,
                                      "R_i_iama_base_under_BoN": bon_value(p_iama, rewards[i], N),
                                      "R_i_naive_base_under_BoN": bon_value(p_naive, rewards[i], N)})
                    obj_iama = agg_value(p_iama, rewards, Ns, w) - beta * kl(p_iama, p_ref)
                    obj_naive = agg_value(p_naive, rewards, Ns, w) - beta * kl(p_naive, p_ref)
                    adapt.append({"K": K, "N": N, "m": m, "seed": seed,
                                  "per_criterion": per_i,
                                  "IAMA_objective_of_iama_base": obj_iama,
                                  "IAMA_objective_of_naive_base": obj_naive,
                                  "improvement": obj_iama - obj_naive})
    res["c_adaptation"] = {
        "n_instances": len(adapt),
        "n_instances_iama_better": int(sum(1 for a in adapt if a["improvement"] > 1e-9)),
        "min_improvement": float(min(a["improvement"] for a in adapt)),
        "median_improvement": float(np.median([a["improvement"] for a in adapt])),
        "instances": adapt,
    }

    # ---------- MUTATION on (c): with N=1 the IAMA and naive optima must coincide ----------
    mut = []
    for K in (6, 10):
        for seed in range(5):
            rng = np.random.default_rng(99 * K + seed)
            r0 = rng.uniform(0, 1, K)
            rewards = [r0, 1.0 - r0]
            w = np.array([0.5, 0.5])
            beta = 0.05
            p_ref = np.full(K, 1.0 / K)
            p_iama = train(rewards, [1, 1], w, beta, p_ref)
            p_naive = train_naive(rewards, w, beta, p_ref)
            mut.append(float(0.5 * np.abs(p_iama - p_naive).sum()))
    res["MUTATION_N1_tv_between_iama_and_naive_optimum"] = {"max": float(max(mut)), "all": mut}

    # sanity: independent solver agrees with the mirror-descent trained policy on one instance
    rng = np.random.default_rng(5)
    K = 8
    r0 = rng.uniform(0, 1, K)
    rewards = [r0, 1 - r0]
    p_ref = np.full(K, 1.0 / K)
    p_md = train(rewards, [4, 4], np.array([0.5, 0.5]), 0.05, p_ref)
    p_lb, _ = solve_optimum(rewards, [4, 4], np.array([0.5, 0.5]), 0.05, p_ref)
    res["sanity_tv_mirror_descent_vs_lbfgs"] = float(0.5 * np.abs(p_md - p_lb).sum())

    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "c_adaptation"}, indent=2)[:3000])
    print("c_adaptation summary:", {k: v for k, v in res["c_adaptation"].items() if k != "instances"})


if __name__ == "__main__":
    main()
