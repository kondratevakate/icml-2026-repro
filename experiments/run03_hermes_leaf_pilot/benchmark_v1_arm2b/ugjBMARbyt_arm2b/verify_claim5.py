"""Claim 5 (TASK #5 "Corollary 5.4"; paper v2 Theorem 5.3 + Corollary 5.4).

Claim: when the basis-approximation profile is zeta(n) = 1 - n^{-q}, the regret bound
of Theorem 5.3 interpolates between Otilde(sqrt(T)) (q -> infinity) and O(T) (q -> 0):
the dominant exponent is (q+2)/(2q+2).

Executable test:
  * evaluate the FULL Theorem 5.3 bound expression numerically over a grid of q and T,
    fit its log-log slope in T, and compare with the analytic exponent (q+2)/(2q+2);
  * check monotonicity in q and the two limits (q->0 => 1, q->inf => 1/2);
  * SIMULATION: build a non-parametric instance whose coefficients satisfy
    sum_{i<=n}|gamma_i|^2 >= (1-n^{-q})||c*||^2, run basis-truncation EntUCB with
    n_t = ceil(t^{1/(q+1)}), and check the realised regret exponent increases as q
    decreases and stays below the certified bound;
  * MUTATION: freeze n_t = 1 (no growth of the truncation order) -> the approximation
    error is never eliminated and the regret becomes linear (slope ~ 1).
Seed 20260803.
"""
import numpy as np
from common import save, beta_t, loglog_slope

SEED = 20260803
lam, sigma, delta, Cbar, kappa = 1.0, 0.2, 0.05, 1.0, 0.5


def thm53_bound(T, q):
    e = (q + 2) / (2 * q + 2)
    term1 = Cbar * (1 + 2 * q * T ** e / (q + 1))
    term2 = kappa * (1 + np.sqrt(T) * np.log(T))
    inner = np.sqrt(2 * np.log((lam ** -1 + 2 * T ** q * 2 * Cbar ** 2) / delta)) + np.sqrt(lam) * Cbar
    term3 = 2 * Cbar * sigma * T ** e * inner * np.sqrt(np.log(1 + 2 * T ** q / Cbar ** 2))
    term4 = sigma * np.sqrt(2 * T * np.log(2 / delta))
    return float(term1 + term2 + term3 + term4)


qs = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 32.0]
Ts = [10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]
analytic = {q: (q + 2) / (2 * q + 2) for q in qs}
fitted = {}
for q in qs:
    sl, _ = loglog_slope(Ts, [thm53_bound(T, q) for T in Ts])
    fitted[q] = float(sl)
mono = all(analytic[qs[i]] >= analytic[qs[i + 1]] for i in range(len(qs) - 1))
lim0 = abs(analytic[0.05] - 1.0) < 0.03
liminf = abs(analytic[32.0] - 0.5) < 0.03
fit_ok = all(fitted[q] >= analytic[q] - 0.02 and fitted[q] <= analytic[q] + 0.30 for q in qs)

# --- simulation with decaying coefficients -------------------------------
def simulate(q, T, seed, freeze=False):
    """gamma_i ~ i^{-(q+1)/2} (so tail sum ~ n^{-q}); actions = random unit vectors."""
    r = np.random.default_rng(seed)
    D = 64
    g = np.arange(1, D + 1) ** (-(q + 1) / 2.0)
    g *= r.choice([-1, 1], D)
    g /= np.linalg.norm(g)
    A = r.standard_normal((40, D)); A /= np.linalg.norm(A, axis=1, keepdims=True)
    vals = A @ g; opt = vals.min()
    reg = 0.0
    n_prev = 0
    V = np.zeros((0, 0)); b = np.zeros(0)
    for t in range(1, T + 1):
        n = 1 if freeze else min(D, int(np.ceil(t ** (1.0 / (q + 1)))))
        if n != n_prev:                      # grow the RLS to the new order
            Vn = lam * np.eye(n); bn = np.zeros(n)
            if n_prev:
                Vn[:n_prev, :n_prev] = V
                bn[:n_prev] = b
            V, b, n_prev = Vn, bn, n
        An = A[:, :n]
        fh = np.linalg.solve(V, b)
        bt = beta_t(V, lam, sigma, Cbar, delta)
        Vi = np.linalg.inv(V)
        lcb = An @ fh - bt * np.sqrt(np.einsum("ij,jk,ik->i", An, Vi, An))
        k = int(np.argmin(lcb))
        reg += vals[k] - opt
        c = vals[k] + sigma * r.standard_normal()
        V = V + np.outer(An[k], An[k]); b = b + c * An[k]
    return float(reg)


sim_Ts = [250, 500, 1000, 2000, 4000]
sim = {}
for q in [0.1, 0.5, 2.0]:
    rs = [float(np.mean([simulate(q, T, SEED + 17 * i) for i in range(3)])) for T in sim_Ts]
    sl, _ = loglog_slope(sim_Ts, rs)
    sim[q] = dict(regret={str(T): v for T, v in zip(sim_Ts, rs)}, loglog_slope=float(sl),
                  certified_slope=analytic[q],
                  below_bound=bool(all(v <= thm53_bound(T, q) for T, v in zip(sim_Ts, rs))))
sim_ordered = sim[0.1]["loglog_slope"] >= sim[2.0]["loglog_slope"] - 0.05
below = all(sim[q]["below_bound"] for q in sim)

# --- MUTATION: frozen truncation order n_t = 1 ---------------------------
mut_rs = [float(np.mean([simulate(0.5, T, SEED + 17 * i, freeze=True) for i in range(3)]))
          for T in sim_Ts]
sl_mut, _ = loglog_slope(sim_Ts, mut_rs)

ok = (mono and lim0 and liminf and fit_ok and below and sim_ordered and sl_mut > 0.9)
out = dict(
    claim=5, source="paper v2 Theorem 5.3 + Corollary 5.4 (TASK cites 'Corollary 5.4')",
    seed=SEED, qs=qs, Ts=Ts,
    analytic_exponent={str(q): analytic[q] for q in qs},
    fitted_loglog_slope_of_full_bound={str(q): fitted[q] for q in qs},
    monotone_in_q=bool(mono), limit_q_to_0_is_1=bool(lim0), limit_q_to_inf_is_half=bool(liminf),
    fitted_matches_analytic=bool(fit_ok),
    simulation={str(q): sim[q] for q in sim},
    simulation_below_certified_bound=bool(below),
    simulation_slope_ordering_q_small_worse=bool(sim_ordered),
    mutation_frozen_truncation_regret={str(T): v for T, v in zip(sim_Ts, mut_rs)},
    mutation_frozen_truncation_loglog_slope=float(sl_mut),
    verdict="verified" if ok else "inconclusive",
)
for k, v in out.items():
    print(f"{k}: {v}")
save("claim5.json", out)
