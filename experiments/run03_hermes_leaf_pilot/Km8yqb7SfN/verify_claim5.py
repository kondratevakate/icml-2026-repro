"""verify_claim5.py — Proposition 3.1 (Section 3.1) of arXiv 2602.01603.

Claim: with Y=[0,1], r1(y)=1-y^2, r2(y)=1-(1-y)^2, w1=w2=0.5, beta=0:
  (a) naive weighted-sum optimum collapses to delta_{0.5};
  (b) IAMA optimum with BoN_N (N>=2) is
        pi*(y) = alpha y^(alpha-1)(1-y)^(alpha-1) / (y^alpha+(1-y)^alpha)^2,  alpha = 1/(N-1).

Tests
  T1 normalization: closed-form CDF F(y)=y^a/(y^a+(1-y)^a) has F'=pi* (sympy) and F(1)-F(0)=1.
  T2 first-order optimality (continuous): for beta=0 the interior maximiser of a linear-in-perturbation
     functional over the simplex must have constant functional derivative on the support:
        0.5*dR1/dpi(y) + 0.5*dR2/dpi(y) = const.
     With Prop. 4.2 and r1 decreasing / r2 increasing in y:
        dR1/dpi(y) = -int_0^y 2N z (1-F(z))^(N-1) dz
        dR2/dpi(y) = -int_y^1 2N (1-z) F(z)^(N-1) dz
     evaluated at F = closed-form CDF. Reported as relative span of the aggregate derivative.
  T3 discrete optimisation: maximise the exact discrete IAMA objective on a K-bin grid by exact
     mirror ascent (Algorithm 1 with beta=0) from several inits; compare to the closed-form bin masses.
  T4 naive optimum: maximise 0.5 E_p[r1] + 0.5 E_p[r2] on the same grid -> mass concentrates at y=0.5.
  MUTATION M1: N=2 (alpha=1) must give the uniform density.
  MUTATION M2: use a wrong alpha (alpha' = 1/N instead of 1/(N-1)); the constant-derivative
     optimality condition of T2 must then be violated by a large margin.

Run: .venv/bin/python verify_claim5.py
"""
import json
import sys
import numpy as np

sys.path.insert(0, ".")
from iama_core import bon_value, bon_grad, linear_value  # noqa: E402

OUT = "results/claim5.json"
CMD = ".venv/bin/python verify_claim5.py"


def pi_star(y, alpha):
    return alpha * y ** (alpha - 1) * (1 - y) ** (alpha - 1) / (y ** alpha + (1 - y) ** alpha) ** 2


def cdf_star(y, alpha):
    return y ** alpha / (y ** alpha + (1 - y) ** alpha)


def agg_deriv_continuous(alpha, N, M=200001):
    """0.5 dR1/dpi(y) + 0.5 dR2/dpi(y) on a fine grid, at pi = closed-form with exponent alpha."""
    z = np.linspace(0.0, 1.0, M)
    F = np.where(z <= 0, 0.0, np.where(z >= 1, 1.0, cdf_star(np.clip(z, 1e-300, 1 - 1e-16), alpha)))
    g1 = 2 * N * z * (1 - F) ** (N - 1)
    g2 = 2 * N * (1 - z) * F ** (N - 1)
    dz = z[1] - z[0]
    c1 = np.concatenate(([0.0], np.cumsum(0.5 * (g1[1:] + g1[:-1])) * dz))       # int_0^y g1
    tot2 = np.sum(0.5 * (g2[1:] + g2[:-1])) * dz
    c2 = tot2 - np.concatenate(([0.0], np.cumsum(0.5 * (g2[1:] + g2[:-1])) * dz))  # int_y^1 g2
    d1 = -c1
    d2 = -c2
    agg = 0.5 * d1 + 0.5 * d2
    inner = (z > 0.02) & (z < 0.98)   # avoid quadrature error at the integrable endpoint singularity
    return agg, z, inner


def mirror_ascent_grid(rewards, N, K=400, iters=6000, eta=2.0, seed=0, use_bon=True):
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, 1, K + 1)
    y = 0.5 * (edges[:-1] + edges[1:])
    rs = [f(y) for f in rewards]
    p = rng.dirichlet(np.full(K, 1.0)) if seed else np.full(K, 1.0 / K)
    for _ in range(iters):
        if use_bon:
            d = 0.5 * bon_grad(p, rs[0], N) + 0.5 * bon_grad(p, rs[1], N)
        else:
            d = 0.5 * rs[0] + 0.5 * rs[1]
        lz = np.log(p) + eta * d
        lz -= lz.max()
        p = np.exp(lz)
        p /= p.sum()
    val = (0.5 * bon_value(p, rs[0], N) + 0.5 * bon_value(p, rs[1], N)) if use_bon else \
          (0.5 * linear_value(p, rs[0]) + 0.5 * linear_value(p, rs[1]))
    return y, edges, p, val


def main():
    res = {"command": CMD, "claim": 5,
           "source": "Proposition 3.1, Section 3.1, arXiv 2602.01603"}
    r1 = lambda y: 1 - y ** 2
    r2 = lambda y: 1 - (1 - y) ** 2

    # ---- T1 symbolic normalization ----
    import sympy as sp
    ys, a = sp.symbols("y alpha", positive=True)
    Fs = ys ** a / (ys ** a + (1 - ys) ** a)
    dens = sp.simplify(sp.diff(Fs, ys) - a * ys ** (a - 1) * (1 - ys) ** (a - 1) / (ys ** a + (1 - ys) ** a) ** 2)
    res["T1_cdf_derivative_matches_density_symbolically"] = bool(sp.simplify(dens) == 0)
    res["T1_total_mass_F1_minus_F0"] = 1.0  # F(1)=1, F(0)=0 by inspection of the closed form
    # numeric cross-check of the mass by quadrature
    from scipy.integrate import quad
    res["T1_numeric_mass"] = {}
    for N in (2, 4, 8, 16):
        al = 1.0 / (N - 1)
        eps = 1e-9
        m, _ = quad(lambda t: pi_star(t, al), eps, 1 - eps, limit=500, points=[0.5])
        # analytic mass of the two excluded end slivers via the closed-form CDF
        m += cdf_star(eps, al) + (1.0 - cdf_star(1 - eps, al))
        res["T1_numeric_mass"][str(N)] = float(m)

    # ---- T2 first-order optimality of the closed form ----
    res["T2_relative_span_of_aggregate_derivative"] = {}
    for N in (2, 3, 4, 8, 16):
        al = 1.0 / (N - 1)
        agg, z, inner = agg_deriv_continuous(al, N)
        v = agg[inner]
        rel = float((v.max() - v.min()) / max(abs(v).max(), 1e-30))
        res["T2_relative_span_of_aggregate_derivative"][str(N)] = rel

    # ---- MUTATION M2: wrong alpha ----
    res["M2_wrong_alpha_relative_span"] = {}
    for N in (3, 4, 8, 16):
        al_bad = 1.0 / N
        agg, z, inner = agg_deriv_continuous(al_bad, N)
        v = agg[inner]
        res["M2_wrong_alpha_relative_span"][str(N)] = float((v.max() - v.min()) / max(abs(v).max(), 1e-30))

    # ---- T3 discrete mirror-ascent optimum vs closed form ----
    res["T3_grid_optimum_vs_closed_form"] = {}
    K = 400
    for N in (2, 4, 8):
        al = 1.0 / (N - 1)
        per_seed = []
        for seed in (0, 1, 2, 3, 4):
            y, edges, p, val = mirror_ascent_grid([r1, r2], N, K=K, seed=seed)
            pref = np.diff(cdf_star(np.clip(edges, 1e-300, 1 - 1e-16), al))
            pref = np.clip(pref, 0, None); pref /= pref.sum()
            tv = float(0.5 * np.abs(p - pref).sum())
            rs = [r1(y), r2(y)]
            val_ref = 0.5 * bon_value(pref, rs[0], N) + 0.5 * bon_value(pref, rs[1], N)
            per_seed.append({"seed": seed, "tv_distance": tv,
                             "objective_mirror_ascent": float(val),
                             "objective_closed_form": float(val_ref),
                             "objective_gap": float(val_ref - val)})
        res["T3_grid_optimum_vs_closed_form"][str(N)] = {
            "per_seed": per_seed,
            "max_tv_distance": max(d["tv_distance"] for d in per_seed),
            "max_abs_objective_gap": max(abs(d["objective_gap"]) for d in per_seed),
        }

    # ---- MUTATION M1: N=2 -> uniform ----
    al = 1.0 / (2 - 1)
    yy = np.linspace(0.001, 0.999, 999)
    res["M1_N2_max_abs_density_minus_1"] = float(np.max(np.abs(pi_star(yy, al) - 1.0)))

    # ---- T4 naive optimum collapses to y=0.5 ----
    y, edges, p, val = mirror_ascent_grid([r1, r2], N=2, K=K, seed=0, use_bon=False, eta=50.0, iters=20000)
    j = int(np.argmax(p))
    res["T4_naive"] = {"argmax_bin_center": float(y[j]), "mass_at_argmax": float(p[j]),
                       "mass_within_0.01_of_0.5": float(p[np.abs(y - 0.5) < 0.01].sum()),
                       "mean": float(np.dot(p, y)), "std": float(np.sqrt(np.dot(p, (y - np.dot(p, y)) ** 2)))}
    # bimodality contrast: IAMA N=8 mass in the tails vs naive
    y8, e8, p8, _ = mirror_ascent_grid([r1, r2], 8, K=K, seed=0)
    res["T4_iama_N8_mass_outside_center"] = float(p8[(y8 < 0.25) | (y8 > 0.75)].sum())
    res["T4_naive_mass_outside_center"] = float(p[(y < 0.25) | (y > 0.75)].sum())

    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
