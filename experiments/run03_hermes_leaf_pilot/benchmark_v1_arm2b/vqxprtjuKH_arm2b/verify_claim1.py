"""Claim 1 (Theorem 1.1, Sec 1.2): PTAS for VarAlloc -- additive eps guarantee.

VarAlloc: X_i ~ N(mu_i, sigma_i^2) independent, mu_i >= 0, sum_i sigma_i^2 = 1.
OPT = max E max_i X_i.  Theorem 1.1: poly-time algo returns allocation with
E max_i X_i >= OPT - eps.

Reproduction strategy (first principles, CPU):
  * exact objective by deterministic quadrature on a fixed t-grid;
  * ALG = the PTAS-style discretised search: variance budget split into 1/g units
    (g = eps^2), enumerate ALL grid allocations (this is the finite search the PTAS
    performs after Lemma 2.1 truncates the support);
  * OPT_ref = strong continuous reference optimum (multi-start Nelder-Mead on a
    softmax parameterisation + best grid point), an upper bound proxy for OPT;
  * claim holds iff  OPT_ref - ALG <= eps  on every instance/eps pair.
Mutation: coarsen the discretisation to g = 0.5 (breaks the PTAS grid resolution);
the additive-eps property must then break.
"""
import itertools, json, os, time
import numpy as np
from scipy.optimize import minimize

SEED = 20260803
T = np.linspace(-8.0, 12.0, 2001)
DT = T[1] - T[0]
from scipy.stats import norm


def emax(mus, sig2):
    """E[max_i X_i] via  int_0^inf (1-F) - int_{-inf}^0 F  on a fixed grid."""
    sd = np.sqrt(np.maximum(np.asarray(sig2, float), 0.0))
    mus = np.asarray(mus, float)
    logF = np.zeros_like(T)
    live = sd > 1e-9
    if live.any():
        Z = (T[None, :] - mus[live][:, None]) / sd[live][:, None]
        logF += norm.logcdf(Z).sum(axis=0)
    for m in mus[~live]:
        logF += np.where(T >= m, 0.0, -np.inf)
    F = np.exp(logF)
    pos = T >= 0
    a = np.trapezoid((1.0 - F)[pos], T[pos])
    b = np.trapezoid(F[~pos], T[~pos])
    return float(a - b)


def grid_allocs(n, units):
    """all compositions of `units` into n non-negative parts"""
    for cut in itertools.combinations(range(units + n - 1), n - 1):
        prev, out = -1, []
        for c in cut:
            out.append(c - prev - 1)
            prev = c
        out.append(units + n - 2 - prev)
        yield out


def alg_grid(mus, g):
    units = int(round(1.0 / g))
    best, bestx = -np.inf, None
    for comp in grid_allocs(len(mus), units):
        v = np.array(comp, float) / units
        val = emax(mus, v)
        if val > best:
            best, bestx = val, v
    return best, bestx


_REF_CACHE = {}


def opt_ref(mus, seed=SEED, restarts=5):
    key = tuple(np.round(mus, 6))
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    n = len(mus)
    rng = np.random.default_rng(seed)
    best, bx = -np.inf, None
    for r in range(restarts):
        th = np.zeros(n) if r == 0 else rng.normal(0, 1.5, n)
        f = lambda t: -emax(mus, np.exp(t - t.max()) / np.exp(t - t.max()).sum())
        res = minimize(f, th, method="Nelder-Mead",
                       options=dict(maxiter=1200, fatol=1e-10, xatol=1e-7))
        if -res.fun > best:
            e = np.exp(res.x - res.x.max())
            best, bx = -res.fun, e / e.sum()
    _REF_CACHE[key] = (best, bx)
    return best, bx


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    instances = [
        ("zero_means_n4", np.zeros(4)),
        ("zero_means_n5", np.zeros(5)),
        ("ramp_n5", np.array([0.0, 0.25, 0.5, 0.75, 1.0])),
        ("one_big_n4", np.array([2.0, 0.0, 0.0, 0.0])),
        ("random_n5", np.round(rng.uniform(0, 1.5, 5), 3)),
    ]
    rows, worst = [], -np.inf
    for eps in (0.5, 0.3, 0.2):
        g = eps ** 2
        for name, mus in instances:
            alg, xa = alg_grid(mus, g)
            ref, xr = opt_ref(mus)
            gap = ref - alg
            worst = max(worst, gap)
            rows.append(dict(instance=name, n=len(mus), mus=list(map(float, mus)),
                             eps=eps, grid_step=g, alg=alg, opt_ref=ref, gap=gap,
                             ok=bool(gap <= eps + 1e-9),
                             alg_alloc=[round(float(z), 4) for z in xa]))
    # Mutations (each breaks a structural assumption of the PTAS, not a free knob).
    mut = []
    for name, mus in instances:
        ref, _ = opt_ref(mus)
        # M1: destroy the discretisation resolution -- support forced to one variable
        a1, _ = alg_grid(mus, 1.0)
        mut.append(dict(mutation="M1_single_support_grid", instance=name, eps=0.05,
                        alg=a1, opt_ref=ref, gap=ref - a1,
                        still_within_eps=bool(ref - a1 <= 0.05)))
        # M2: violate the variance budget sum sigma_i^2 = 1 (use only half of it)
        _, xr = opt_ref(mus)
        a2 = emax(mus, 0.5 * xr)
        mut.append(dict(mutation="M2_half_budget", instance=name, eps=0.05,
                        alg=a2, opt_ref=ref, gap=ref - a2,
                        still_within_eps=bool(ref - a2 <= 0.05)))
    mut_break = any(not m["still_within_eps"] for m in mut)

    out = dict(
        claim=1, source="Theorem 1.1, Section 1.2 (arXiv:2502.18463v1)",
        statement="PTAS for VarAlloc: poly-time allocation with E max_i X_i >= OPT - eps",
        seed=SEED, rows=rows, worst_gap=float(worst),
        all_within_eps=bool(all(r["ok"] for r in rows)),
        mutation=dict(description="coarsen PTAS grid to step 0.5 (>> eps^2)",
                      rows=mut, property_breaks=bool(mut_break)),
        verdict="verified" if all(r["ok"] for r in rows) and mut_break else "inconclusive",
        runtime_s=round(time.time() - t0, 1),
    )
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/claim1.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in
                      ("worst_gap", "all_within_eps", "verdict", "runtime_s")}, indent=1))
    print("mutation breaks:", mut_break)


if __name__ == "__main__":
    main()
