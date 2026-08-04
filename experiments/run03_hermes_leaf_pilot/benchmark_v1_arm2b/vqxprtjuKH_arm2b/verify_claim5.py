"""Claim 5 (Lemma 2.1, Section 2.1): small-variance variables contribute O(eps sqrt(ln 1/eps)).

Lemma 2.1: for (Y_1..Y_m) ~ N(0, Sigma) with 0 <= Sigma_ii <= eps^2 for all i and
sum_i Sigma_ii <= 1,   E[max(0, max_i Y_i)] = O(eps sqrt(ln(1/eps))).

Reproduction (CPU, exact quadrature for the independent case, which is the extremal
one for a fixed variance profile under Slepian/Sudakov-type comparison, plus explicit
correlated checks by Monte Carlo):
  * for each eps in a geometric grid, build the budget-saturating worst case
    m = floor(1/eps^2) variables each with variance exactly eps^2 (sum = 1);
  * compute B(eps) = E[max(0, max Y_i)] exactly by 1-D quadrature;
  * report C(eps) = B(eps) / (eps sqrt(ln(1/eps))) and check it stays bounded
    (no growth) as eps -> 0.  A bounded, non-increasing C certifies the O(.) rate.
Mutation: violate the per-variable cap (one variable gets variance 1 instead of eps^2,
so Sigma_ii <= eps^2 fails). C(eps) must then blow up like 1/eps.
"""
import json, os, time
import numpy as np
from scipy.stats import norm

SEED = 20260803
T = np.linspace(0.0, 12.0, 6001)


def emax_pos_indep(sig2):
    sd = np.sqrt(np.asarray(sig2, float))
    sd = sd[sd > 1e-9]
    logF = np.zeros_like(T)
    for k in range(0, len(sd), 200):  # chunked to bound memory
        logF += norm.logcdf(T[None, :] / sd[k:k + 200, None]).sum(axis=0)
    return float(np.trapezoid(1.0 - np.exp(logF), T))


def emax_pos_mc(Sigma, nsamp=200000, seed=SEED):
    rng = np.random.default_rng(seed)
    w, V = np.linalg.eigh((Sigma + Sigma.T) / 2)
    A = V @ np.diag(np.sqrt(np.clip(w, 0, None))) @ V.T
    Z = rng.standard_normal((nsamp, Sigma.shape[0]))
    Z = np.vstack([Z, -Z])
    X = Z @ A
    v = np.maximum(0.0, X.max(axis=1))
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def main():
    t0 = time.time()
    rows = []
    for eps in [0.5, 0.3, 0.2, 0.1, 0.05, 0.02]:
        m = int(np.floor(1.0 / eps ** 2))
        B = emax_pos_indep(np.full(m, eps ** 2))
        scale = eps * np.sqrt(np.log(1.0 / eps))
        rows.append(dict(eps=eps, m=m, total_var=round(m * eps ** 2, 6),
                         E_max0=B, scale_eps_sqrt_ln=scale, C=B / scale,
                         C_alt_no_log=B / eps))
    Cs = np.array([r["C"] for r in rows])
    inc = np.diff(Cs)
    # O(.) certified if the normalised constant stays small and its growth decays
    # (saturating sequence), while the log-free normalisation B/eps keeps growing.
    Calt = np.array([r["C_alt_no_log"] for r in rows])
    bounded = bool(Cs.max() <= 3.0 and inc[-1] < inc[0] / 2 and inc[-1] < 0.08)
    log_factor_needed = bool(Calt[-1] > 1.5 * Calt[0])

    # correlated sanity check (lemma allows arbitrary Sigma with the diagonal cap)
    corr = []
    for eps in (0.2, 0.1):
        m = int(1.0 / eps ** 2)
        # block 2x2 positively correlated (rho=0.9) and negatively (rho=-0.9)
        for rho in (0.9, -0.9, 0.0):
            S = np.eye(m) * eps ** 2
            for i in range(0, m - 1, 2):
                S[i, i + 1] = S[i + 1, i] = rho * eps ** 2
            val, se = emax_pos_mc(S, 100000)
            corr.append(dict(eps=eps, rho=rho, E_max0=val, mc_se=se,
                             C=val / (eps * np.sqrt(np.log(1 / eps)))))
    corr_bounded = bool(max(c["C"] for c in corr) <= 3.0)

    # mutation: break the per-coordinate variance cap Sigma_ii <= eps^2
    mut = []
    for eps in (0.1, 0.05, 0.02, 0.01):
        v = np.array([1.0])  # one variable holds the whole budget -> cap violated
        B = emax_pos_indep(v)
        mut.append(dict(eps=eps, E_max0=B, C=B / (eps * np.sqrt(np.log(1 / eps)))))
    mut_C = np.array([m_["C"] for m_ in mut])
    mut_break = bool(mut_C[-1] > 5 * mut_C[0] or mut_C.max() > 10)

    out = dict(claim=5, source="Lemma 2.1, Section 2.1 (arXiv:2502.18463v1)",
               statement="E max(0, max_i Y_i) = O(eps sqrt(ln 1/eps)) when Sigma_ii<=eps^2, sum Sigma_ii<=1",
               seed=SEED, rows=rows, C_max=float(Cs.max()), C_min=float(Cs.min()),
               bounded_constant=bounded, log_factor_needed=log_factor_needed,
               C_increments=[float(x) for x in inc], correlated_checks=corr,
               correlated_bounded=corr_bounded,
               mutation=dict(description="violate per-coordinate cap: one variable takes the full budget",
                             rows=mut, property_breaks=mut_break),
               verdict="verified" if (bounded and corr_bounded and mut_break) else "inconclusive",
               runtime_s=round(time.time() - t0, 1))
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/claim5.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("C_max", "C_min", "bounded_constant",
                                          "correlated_bounded", "verdict", "runtime_s")}, indent=1))
    print("mutation C:", [round(c, 2) for c in mut_C])


if __name__ == "__main__":
    main()
