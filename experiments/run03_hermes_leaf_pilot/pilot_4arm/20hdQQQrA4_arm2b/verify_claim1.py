"""Claim 1 (Theorem 3.5): approximation-error bound ||P* - f_t||_p < (3 + 3 sqrt(n_out)) K.

Strategy (CPU, analytic + exhaustive numeric):
  A) Symbolic check that the proof's constant closes: K = eps/(3+3 sqrt(n_out))
     => (3+3 sqrt(n_out)) K == eps exactly (sympy).
  B) Numeric stress test of the theorem's conclusion. For each (n_out, m) and many seeds:
     draw a feasible polyhedron S, a feasible target f_t in S, an "unconstrained network"
     output f_theta with ||f_theta - f_t||_p < K, a null-space net output w_phi with
     ||w_phi||_p < 2K (as the proof bounds it), run the real CAffNet (Eq 12) and record
     ratio = ||P* - f_t||_p / K. The theorem requires ratio < 3 + 3 sqrt(n_out).
  C) Mutation test: replace the claimed constant by a smaller one (1 + sqrt(n_out)) and
     also mutate the algorithm (pick the WORST feasible candidate instead of argmin)
     and show the observed max ratio behaves as predicted.

Run: ./.venv/bin/python verify_claim1.py
"""
import json, math, os
import numpy as np
import sympy as sp
from caffnet_core import caffnet, random_feasible_problem, gammas, P_gamma

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "claim1.json")
P = 2


def symbolic_check():
    eps, n = sp.symbols("epsilon n_out", positive=True)
    K = eps / (3 + 3 * sp.sqrt(n))
    return sp.simplify((3 + 3 * sp.sqrt(n)) * K - eps) == 0


def worst_caffnet(f, A, b, w, tol=1e-9):
    """Mutation: choose the FARTHEST feasible candidate instead of argmin (Eq 12 inverted)."""
    m, n = A.shape
    if np.all(A @ f - b <= tol):
        return f.copy()
    best = None
    for g in gammas(m, n):
        y = P_gamma(f, A, b, g, w)
        if np.all(A @ y - b <= tol):
            d = np.linalg.norm(y - f, P)
            if best is None or d > best[0]:
                best = (d, y)
    return None if best is None else best[1]


def run():
    rng_master = np.random.default_rng(0)
    K = 1e-3
    configs = [(1, 2), (1, 4), (2, 3), (2, 5), (3, 4), (3, 7), (4, 6), (5, 8)]
    n_seeds = 200
    rows, mut_rows = [], []
    infeasible = 0
    for n_out, m in configs:
        bound = 3 + 3 * math.sqrt(n_out)
        ratios, mratios, nproj = [], [], 0
        for s in range(n_seeds):
            rng = np.random.default_rng(rng_master.integers(1 << 31))
            A, b, y0 = random_feasible_problem(rng, n_out, m)
            # Put the target ON the boundary (a random subset of constraints active,
            # slack of the rest is O(K)) so that f_theta genuinely violates constraints
            # and the projection branch of Eq (12) is exercised.
            n_active = int(rng.integers(1, min(m, n_out) + 1))
            act = rng.choice(m, size=n_active, replace=False)
            b = A @ y0 + np.abs(rng.normal(size=m)) * K
            b[act] = A[act] @ y0
            f_t = y0                                   # feasible target value (on boundary)
            d = rng.normal(size=n_out); d /= np.linalg.norm(d, P)
            f_theta = f_t + d * K * rng.uniform(0.0, 0.999)      # ||f_th - f_t||_p < K
            dw = rng.normal(size=n_out); dw /= np.linalg.norm(dw, P)
            w_phi = dw * 2 * K * rng.uniform(0.0, 0.999)         # ||w_phi||_p < 2K
            y, info = caffnet(f_theta, A, b, w=w_phi, p=P)
            if info["branch"] == "projected":
                nproj += 1
            if y is None:
                infeasible += 1
                continue
            ratios.append(float(np.linalg.norm(y - f_t, P) / K))
            yw = worst_caffnet(f_theta, A, b, w_phi)
            if yw is not None:
                mratios.append(float(np.linalg.norm(yw - f_t, P) / K))
        rows.append({
            "n_out": n_out, "m": m, "bound_3_plus_3sqrt_n": bound,
            "n_instances": len(ratios), "n_projection_branch": nproj,
            "max_ratio": max(ratios), "mean_ratio": float(np.mean(ratios)),
            "holds": bool(max(ratios) < bound),
            "holds_under_tight_constant_1_plus_sqrt_n": bool(max(ratios) < 1 + math.sqrt(n_out)),
        })
        mut_rows.append({
            "n_out": n_out, "m": m, "mutation": "argmax instead of argmin in Eq 12",
            "max_ratio": max(mratios), "holds": bool(max(mratios) < bound),
        })
    res = {
        "claim": 1,
        "source": "Theorem 3.5 + proof in Appendix C (K = eps/(3+3 sqrt(n_out)))",
        "command": "./.venv/bin/python verify_claim1.py",
        "symbolic_constant_closes": symbolic_check(),
        "K_used": K, "p_norm": P, "seeds_per_config": 200,
        "infeasible_instances": infeasible,
        "per_config": rows,
        "mutation_worst_candidate": mut_rows,
        "n_configs_bound_holds": sum(r["holds"] for r in rows),
        "n_configs_total": len(rows),
        "max_ratio_overall": max(r["max_ratio"] for r in rows),
    }
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    run()
