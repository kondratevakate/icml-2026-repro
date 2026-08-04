"""Claim 2 (Theorem 1.2, Section 1.2): PTAS for CorrVarAlloc -- additive eps guarantee
with correlated Gaussians.

CorrVarAlloc: choose a PSD covariance Sigma with trace(Sigma) = 1 (variance budget),
X ~ N(0, Sigma); OPT = max_Sigma E max_i X_i.  Theorem 1.2: poly-time algorithm
returns Sigma_hat with E max_i X_i >= OPT - eps.

CPU reproduction:
  * objective by Monte Carlo with COMMON RANDOM NUMBERS (fixed antithetic sample,
    fixed seed) so that ALG and the reference optimum are compared on the identical
    estimator; MC standard errors reported;
  * ALG (finite, poly-size search, PTAS-style discretisation): best over
      (a) diagonal Sigma on the variance grid of step eps^2, and
      (b) equicorrelated Sigma (uniform variances, rho on a grid of step eps^2
          within the PSD range [-1/(n-1), 1]);
  * OPT_ref: multi-start Nelder-Mead over an unconstrained Cholesky factor L,
    Sigma = LL^T / trace(LL^T) -- covers ALL feasible PSD Sigma;
  * claim holds iff OPT_ref - ALG <= eps (checked with a 3-sigma MC margin).
Mutation: halve the trace budget (violates trace(Sigma)=1); the additive-eps
property must break.
"""
import itertools, json, os, time
import numpy as np
from scipy.optimize import minimize

SEED = 20260803
NSAMP = 60000


def make_Z(n, seed=SEED):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((NSAMP, n))
    return np.vstack([Z, -Z])


def emax_mc(Sigma, Z):
    w, V = np.linalg.eigh((Sigma + Sigma.T) / 2)
    A = V @ (np.sqrt(np.clip(w, 0, None))[:, None] * V.T)
    v = (Z @ A).max(axis=1)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def grid_allocs(n, units):
    for cut in itertools.combinations(range(units + n - 1), n - 1):
        prev, out = -1, []
        for c in cut:
            out.append(c - prev - 1)
            prev = c
        out.append(units + n - 2 - prev)
        yield np.array(out, float) / units


def alg(n, eps, Z):
    units = int(round(1.0 / eps ** 2))
    best, bs = -np.inf, None
    for v in grid_allocs(n, units):
        val, _ = emax_mc(np.diag(v), Z)
        if val > best:
            best, bs = val, np.diag(v)
    lo = -1.0 / (n - 1)
    for rho in np.arange(lo, 1.0 + 1e-9, eps ** 2):
        S = np.full((n, n), rho / n) + np.eye(n) * (1 - rho) / n
        val, _ = emax_mc(S, Z)
        if val > best:
            best, bs = val, S
    return best, bs


def opt_ref(n, Z, restarts=8, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.tril_indices(n)

    def build(theta):
        L = np.zeros((n, n))
        L[idx] = theta
        S = L @ L.T
        tr = np.trace(S)
        return S / tr if tr > 1e-12 else np.eye(n) / n

    best, bs = -np.inf, None
    for r in range(restarts):
        th0 = (np.eye(n)[idx] if r == 0 else rng.normal(0, 1, len(idx[0])))
        f = lambda th: -emax_mc(build(th), Z)[0]
        res = minimize(f, th0, method="Nelder-Mead",
                       options=dict(maxiter=6000, fatol=1e-9, xatol=1e-6))
        if -res.fun > best:
            best, bs = -res.fun, build(res.x)
    return best, bs


def main():
    t0 = time.time()
    rows = []
    for n in (3, 4):
        Z = make_Z(n)
        ref, Sref = opt_ref(n, Z)
        _, se = emax_mc(Sref, Z)
        for eps in (0.3, 0.2):
            a, Sa = alg(n, eps, Z)
            gap = ref - a
            rows.append(dict(n=n, eps=eps, alg=a, opt_ref=ref, gap=gap, mc_se=se,
                             ok=bool(gap <= eps + 3 * se),
                             ok_strict=bool(gap + 3 * se <= eps),
                             opt_ref_Sigma=[[round(float(x), 4) for x in r] for r in Sref],
                             alg_Sigma=[[round(float(x), 4) for x in r] for r in Sa]))
        # mutation: halve the trace budget
        am, _ = alg(n, 0.2, Z)
        half, _ = emax_mc(0.5 * Sref, Z)
        rows.append(dict(n=n, eps=0.2, mutation="half_trace_budget",
                         alg=half, opt_ref=ref, gap=ref - half, mc_se=se,
                         ok=bool(ref - half <= 0.2)))
    main_rows = [r for r in rows if "mutation" not in r]
    mut_rows = [r for r in rows if "mutation" in r]
    all_ok = all(r["ok"] for r in main_rows)
    mut_break = any(not r["ok"] for r in mut_rows)
    out = dict(claim=2, source="Theorem 1.2, Section 1.2 (arXiv:2502.18463v1)",
               statement="PTAS for CorrVarAlloc: E max_i X_i >= OPT - eps with correlated Gaussians",
               seed=SEED, mc_samples=2 * NSAMP, rows=rows,
               worst_gap=float(max(r["gap"] for r in main_rows)),
               all_within_eps=bool(all_ok),
               mutation=dict(description="halve trace(Sigma) budget", rows=mut_rows,
                             property_breaks=bool(mut_break)),
               verdict="verified" if (all_ok and mut_break) else "inconclusive",
               runtime_s=round(time.time() - t0, 1))
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/claim2.json", "w"), indent=1)
    for r in rows:
        print(r.get("mutation", "main"), "n", r["n"], "eps", r["eps"],
              "alg", round(r["alg"], 4), "ref", round(r["opt_ref"], 4),
              "gap", round(r["gap"], 4), "se", round(r["mc_se"], 5), "ok", r["ok"])
    print("verdict", out["verdict"], out["runtime_s"], "s")


if __name__ == "__main__":
    main()
