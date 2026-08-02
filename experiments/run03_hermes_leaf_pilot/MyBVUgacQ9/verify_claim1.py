"""verify_claim1.py -- CLAIM 1 / Theorem 3.1 (Section 3).

Claim: under Assumptions 2.3 and 2.6 and rho >= max{4 L_phi^2/(mu_phi lmin+(Q^T Q)),
4 L_phi^2/(mu_phi sqrt(lmin+(Q Q^T)))}, ADMM (Algorithm 1) converges with at least
sublinear rate, L^k - L* in o(1/k), to a stationary point that is blockwise optimal.

Tests (exhaustive over an initialisation grid, never a single seed):
  T1  convergence + feasibility of the limit point
  T2  o(1/k): k * (L^k - L*) -> 0
  T3  blockwise (Nash-like) optimality of the limit point
  MUT-A  violate Assumption 2.6 (Example 2.8: min x^2+y^2 s.t. xy=1, no z-block,
         Q = 0 -> not full row rank): ADMM must FAIL (w -> -inf, infeasible limit)
  MUT-B  rho far below the Theorem-3.1 threshold on a problem where the threshold bites.

Run:  .venv/bin/python verify_claim1.py
"""
import itertools
import json
import time

import numpy as np

import admm_lib as A

OUT = "results/claim1.json"
CMD = ".venv/bin/python verify_claim1.py"


def blockwise_optimality(prob, x, z, mu_x):
    """For each block i, the exact minimiser of (mu/2)||x_i||^2 subject to
    A(x_i; x_-i) + Q z = 0 (Theorem 3.1's characterisation of the limit point)."""
    errs = []
    for i, b in enumerate(prob.blocks):
        M, c = prob._affine_in_block(x, b)
        rhs = -(c + prob.Q @ z)
        D = np.sqrt(mu_x[b])
        Ms = M / D[None, :]
        y, *_ = np.linalg.lstsq(Ms, rhs, rcond=None)
        cand = y / D
        if np.linalg.norm(Ms @ y - rhs) > 1e-8:
            errs.append(float("nan"))  # subproblem infeasible -> skip
        else:
            errs.append(float(np.linalg.norm(cand - x[b])))
    return errs


def main():
    t0 = time.time()
    res = {"claim": 1, "source": "Theorem 3.1, Section 3", "command": CMD, "tests": {}}

    # ---------------- T1-T3 : toy problem + Example 2.2 --------------------
    grid = [-2.0, -0.5, 0.5, 2.0]
    runs = []
    for q in [0.5, 2.0, 5.0]:
        prob = A.toy_problem(q, mu_x=1.0, mu_z=1.0)
        thr = A.rho_threshold(prob, L_phi=1.0, mu_phi=1.0)
        for rho in [thr, 2 * thr, 4.0 + thr]:
            for x0 in itertools.product(grid, repeat=2):
                x0v = np.array([x0[0], x0[1], -x0[0], 0.5 * x0[1]])
                for z0 in [-1.0, 0.0, 1.0]:
                    x, z, w, h = prob.run(x0v, np.array([z0]), np.zeros(1), rho,
                                          iters=1500)
                    L = h["L"]
                    Lstar = float(L[-1])
                    gap = np.abs(L - Lstar)
                    sw = A.sublinear_witness(gap)
                    bo = blockwise_optimality(prob, x, z, prob.mu_x)
                    runs.append({
                        "problem": "toy(q=%g)" % q, "rho": float(rho),
                        "x0": x0v.tolist(), "z0": z0,
                        "final_res": float(h["res"][-1]),
                        "final_viol": float(h["viol"][-1]),
                        "L_star": Lstar,
                        "last_k_gap": sw["last_k_gap"],
                        "max_k_gap_tail": sw["max_k_gap_tail"],
                        "blockwise_opt_err": bo,
                    })

    p22 = A.example22()
    thr22 = A.rho_threshold(p22, L_phi=2.0, mu_phi=2.0)
    for x0 in itertools.product([-2.0, 0.0, 2.0], repeat=2):
        for rho in [thr22, 2 * thr22]:
            x, z, w, h = p22.run(np.array(x0), np.zeros(2), np.zeros(2), rho, iters=1500)
            L = h["L"]
            gap = np.abs(L - L[-1])
            sw = A.sublinear_witness(gap)
            runs.append({
                "problem": "example2.2", "rho": float(rho), "x0": list(x0), "z0": 0.0,
                "final_res": float(h["res"][-1]), "final_viol": float(h["viol"][-1]),
                "L_star": float(L[-1]), "last_k_gap": sw["last_k_gap"],
                "max_k_gap_tail": sw["max_k_gap_tail"],
                "blockwise_opt_err": blockwise_optimality(p22, x, z, p22.mu_x),
            })

    res["tests"]["T1_T3_runs"] = runs
    res["tests"]["summary"] = {
        "n_runs": len(runs),
        "max_final_res": max(r["final_res"] for r in runs),
        "max_final_viol": max(r["final_viol"] for r in runs),
        "max_last_k_gap": max(r["last_k_gap"] for r in runs),
        "max_blockwise_opt_err": float(np.nanmax(
            [np.nanmax(r["blockwise_opt_err"]) for r in runs])),
        "n_converged_res_lt_1e-8": sum(r["final_res"] < 1e-8 for r in runs),
        "n_feasible_viol_lt_1e-8": sum(r["final_viol"] < 1e-8 for r in runs),
        "n_k_gap_lt_1e-6": sum(r["last_k_gap"] < 1e-6 for r in runs),
    }

    # ---------------- MUT-A : violate Assumption 2.6 -----------------------
    # Example 2.8 : min x^2 + y^2 s.t. x y = 1.  Written in the eq.-1 template with an
    # (unused) z-block whose Q = 0  =>  Q is NOT full row rank (Assumption 2.6 fails).
    C = np.zeros((1, 2, 2)); C[0, 0, 1] = C[0, 1, 0] = 1.0
    bad = A.MultiAffineProblem(blocks=[[0], [1]], C=C, d=np.zeros((1, 2)),
                               e=np.array([-1.0]), Q=np.zeros((1, 1)),
                               mu_x=2.0 * np.ones(2), mu_z=2.0 * np.ones(1))
    muta = []
    for x0 in [0.5, 1.0, 2.0, 5.0]:
        for rho in [1.0, 10.0, 100.0]:
            x, z, w, h = bad.run(np.array([x0, 0.0]), np.zeros(1), np.zeros(1), rho,
                                 iters=4000)
            muta.append({"x0": x0, "rho": rho, "x_final": x.tolist(),
                         "w_final": float(w[0]), "viol_final": float(h["viol"][-1]),
                         "norm_x_final": float(np.linalg.norm(x))})
    res["tests"]["MUT_A_assumption26_violated"] = {
        "runs": muta,
        "max_norm_x_final": max(r["norm_x_final"] for r in muta),
        "max_w_final": max(r["w_final"] for r in muta),
        "min_viol_final": min(r["viol_final"] for r in muta),
        "all_collapse_to_origin": all(r["norm_x_final"] < 1e-3 for r in muta),
        "all_dual_diverges_negative": all(r["w_final"] < -1e3 for r in muta),
        "all_infeasible": all(r["viol_final"] > 0.9 for r in muta),
    }

    # ---------------- MUT-B : rho below the Theorem-3.1 threshold ----------
    # phi(z) = (mu_z/2) z^2 with mu_z = L_phi = 5 ; Q = [[0.2]] -> threshold is large.
    mutb = []
    probb = A.toy_problem(0.2, mu_x=1.0, mu_z=5.0)
    thrb = A.rho_threshold(probb, L_phi=5.0, mu_phi=5.0)
    for frac in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 4.0]:
        rho = frac * thrb
        worst_res, worst_viol = 0.0, 0.0
        for s in range(8):
            r = np.random.default_rng(s)
            x, z, w, h = probb.run(r.normal(size=4) * 2, r.normal(size=1),
                                   np.zeros(1), rho, iters=1500)
            worst_res = max(worst_res, float(h["res"][-1]))
            worst_viol = max(worst_viol, float(h["viol"][-1]))
        mutb.append({"rho_over_threshold": frac, "rho": float(rho),
                     "worst_final_res": worst_res, "worst_final_viol": worst_viol})
    res["tests"]["MUT_B_rho_sweep"] = {"threshold": float(thrb), "runs": mutb}

    res["elapsed_sec"] = time.time() - t0
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    s = res["tests"]["summary"]
    print("T1-T3 runs:", s["n_runs"])
    print("  converged (res<1e-8):", s["n_converged_res_lt_1e-8"],
          " feasible (viol<1e-8):", s["n_feasible_viol_lt_1e-8"])
    print("  max final residual %.3e  max final violation %.3e" %
          (s["max_final_res"], s["max_final_viol"]))
    print("  max_last_k_gap (o(1/k) witness) %.3e ; runs with k*gap<1e-6: %d/%d" %
          (s["max_last_k_gap"], s["n_k_gap_lt_1e-6"], s["n_runs"]))
    print("  max blockwise-optimality error %.3e" % s["max_blockwise_opt_err"])
    ma = res["tests"]["MUT_A_assumption26_violated"]
    print("MUT-A (Asm 2.6 violated): collapse->0:", ma["all_collapse_to_origin"],
          " w->-inf:", ma["all_dual_diverges_negative"],
          " infeasible:", ma["all_infeasible"],
          " max||x||=%.2e min viol=%.3f max w=%.1f" %
          (ma["max_norm_x_final"], ma["min_viol_final"], ma["max_w_final"]))
    print("MUT-B rho sweep (threshold %.3g):" % thrb)
    for r in mutb:
        print("   rho/thr=%8.4g  worst res %.2e  worst viol %.2e" %
              (r["rho_over_threshold"], r["worst_final_res"], r["worst_final_viol"]))
    print("wrote", OUT, "in %.1fs" % res["elapsed_sec"])


if __name__ == "__main__":
    main()
