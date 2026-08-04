"""verify_claim2.py -- CLAIM 2 / Theorem 3.2 (Section 3, Equation 4).

Claim: with the assumptions of Thm 3.1, f L_f-smooth and L second-order differentiable
at the limit point, if ||C|| := max_i ||C_i|| satisfies the bound (4)
    ||C|| in O( ||(QQ^T)^{-1}Q||^{-1} min{ m1, m2 lmin(QQ^T)^{1/2},
                                            m3 lmin(QQ^T)^{1/4} ||(QQ^T)^{-1}Q||^{1/2} } )
then L^k - L* in O(c1^{-k}) with c1>1 (LINEAR rate), and (x*,z*) is a local minimum.

Only the *qualitative* content of (4) is testable: the constants m1,m2,m3 are not given
numerically in the paper.  What (4) says operationally is: for a FIXED Q there is a
threshold on ||C|| below which the rate is linear, and ||C|| = 0 (linear constraints)
is always inside the region.  So we sweep ||C|| with Q fixed (and, separately, sweep Q
with C fixed) and measure the asymptotic per-iteration contraction factor.

Design:
  problem: Section-5 toy, min (mu_x/2)||x||^2 + (mu_z/2) z^2
           s.t. s*(x1 x2 - x3 x4) + q z + 1 = 0    ->  ||C|| = s, Q = [q]
  the Lagrangian is C-infinity everywhere here (no indicators), and we check numerically
  that the limit point is a nondegenerate (positive definite reduced Hessian) local min,
  so the second-order-differentiability hypothesis holds.
  Exhaustive over a 4x3 grid of initialisations x 3 rho values (never one seed).

  MUTATION: push ||C|| far above the small-||C|| regime and show the linear rate is
  lost / degraded, while the same runs with ||C||=0 stay linear.

Run:  .venv/bin/python verify_claim2.py
"""
import itertools
import json
import time

import numpy as np

import admm_lib as A

OUT = "results/claim2.json"
CMD = ".venv/bin/python verify_claim2.py"

INITS = [np.array(v, float) for v in itertools.product([-2.0, 0.5], repeat=4)]
Z0S = [-1.0, 0.0, 1.0]


def local_min_certificate(prob, x, z, rho, w):
    """Second-order check at the limit point: reduced Hessian of the augmented
    Lagrangian in (x,z) must be PSD (needed for 'second-order differentiable at the
    limit point' + 'local minimum' part of Thm 3.2)."""
    v = np.concatenate([x, z])
    n = len(v)
    h = 1e-5

    def L(u):
        return prob.lagrangian(u[:len(x)], u[len(x):], w, rho)

    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n); ei[i] = h
            ej = np.zeros(n); ej[j] = h
            H[i, j] = (L(v + ei + ej) - L(v + ei - ej) - L(v - ei + ej) + L(v - ei - ej)) / (4 * h * h)
    H = 0.5 * (H + H.T)
    return float(np.linalg.eigvalsh(H).min())


def sweep(c_scales, q, rhos, iters=1200):
    out = []
    for s in c_scales:
        prob = A.toy_problem(q, mu_x=1.0, mu_z=1.0, c_scale=s)
        cn = prob.cnorm()
        for rho in rhos:
            facs, r2s, mineig, viols = [], [], [], []
            for x0 in INITS:
                for z0 in Z0S:
                    x, z, w, h = prob.run(x0, np.array([z0]), np.zeros(1), rho, iters=iters)
                    if not np.all(np.isfinite(np.concatenate([x, z, w]))):
                        facs.append(float("inf")); r2s.append(0.0)
                        viols.append(float("inf")); continue
                    f, npts, r2 = A.contraction_factor(h["res"], floor=1e-14)
                    if not np.isfinite(f):
                        # converged to machine precision before the fit window:
                        # bound the factor by the observed decay over the usable range
                        r = h["res"]; idx = np.where(r > 1e-14)[0]
                        if len(idx) >= 3:
                            f = float(np.exp((np.log(r[idx[-1]]) - np.log(r[idx[0]])) /
                                             max(1, idx[-1] - idx[0])))
                            r2 = 1.0
                        else:
                            f = 0.0; r2 = 1.0
                    facs.append(f); r2s.append(r2)
                    viols.append(float(h["viol"][-1]))
                    mineig.append(local_min_certificate(prob, x, z, rho, w))
            out.append({
                "c_scale": s, "C_norm": cn, "q": q, "rho": rho,
                "n_inits": len(facs),
                "worst_factor": float(np.max(facs)),
                "median_factor": float(np.median(facs)),
                "min_r2_of_loglinear_fit": float(np.min(r2s)),
                "worst_final_viol": float(np.max(viols)),
                "min_reduced_hessian_eig": float(np.min(mineig)) if mineig else None,
            })
    return out


def main():
    t0 = time.time()
    res = {"claim": 2, "source": "Theorem 3.2 + Equation 4, Section 3", "command": CMD}
    q = 2.0
    rhos = [1.0, 2.0, 4.0]

    small = sweep([0.0, 0.01, 0.05, 0.2, 0.5, 1.0], q, rhos)
    large = sweep([2.0, 5.0, 10.0, 25.0, 50.0], q, rhos)   # MUTATION arm
    res["small_C_regime"] = small
    res["MUTATION_large_C_regime"] = large

    # best (over rho) worst-case-over-initialisations factor, per ||C||
    def best_by_C(rows):
        d = {}
        for r in rows:
            k = r["C_norm"]
            if k not in d or r["worst_factor"] < d[k]["worst_factor"]:
                d[k] = r
        return [d[k] for k in sorted(d)]

    res["profile_small"] = [{"C_norm": r["C_norm"], "best_rho": r["rho"],
                             "worst_factor": r["worst_factor"],
                             "worst_final_viol": r["worst_final_viol"]}
                            for r in best_by_C(small)]
    res["profile_large"] = [{"C_norm": r["C_norm"], "best_rho": r["rho"],
                             "worst_factor": r["worst_factor"],
                             "worst_final_viol": r["worst_final_viol"]}
                            for r in best_by_C(large)]
    res["linear_reference_C0"] = res["profile_small"][0]["worst_factor"]
    res["verdict_numbers"] = {
        "max_worst_factor_over_small_C": max(r["worst_factor"] for r in res["profile_small"]),
        "min_worst_factor_over_large_C": min(r["worst_factor"] for r in res["profile_large"]),
        "all_small_C_linear_factor_lt_1": all(r["worst_factor"] < 0.999
                                              for r in res["profile_small"]),
        "some_large_C_not_linear": any(r["worst_factor"] >= 0.999 or
                                       not np.isfinite(r["worst_factor"])
                                       for r in res["profile_large"]),
        "monotone_degradation": all(
            res["profile_small"][i]["worst_factor"] <= res["profile_small"][i + 1]["worst_factor"] + 1e-9
            for i in range(len(res["profile_small"]) - 1)),
    }
    res["elapsed_sec"] = time.time() - t0
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)

    print("q = %g, rho grid %s, %d initialisations per cell" % (q, rhos, len(INITS) * len(Z0S)))
    print("small-||C|| regime (Thm 3.2 region):")
    for r in res["profile_small"]:
        print("   ||C||=%6.3g  best rho=%4g  worst contraction factor %.4f  viol %.1e"
              % (r["C_norm"], r["best_rho"], r["worst_factor"], r["worst_final_viol"]))
    print("MUTATION large-||C|| regime (eq. 4 violated):")
    for r in res["profile_large"]:
        print("   ||C||=%6.3g  best rho=%4g  worst contraction factor %.4f  viol %.1e"
              % (r["C_norm"], r["best_rho"], r["worst_factor"], r["worst_final_viol"]))
    print(json.dumps(res["verdict_numbers"], indent=1))
    print("wrote", OUT, "in %.1fs" % res["elapsed_sec"])


if __name__ == "__main__":
    main()
