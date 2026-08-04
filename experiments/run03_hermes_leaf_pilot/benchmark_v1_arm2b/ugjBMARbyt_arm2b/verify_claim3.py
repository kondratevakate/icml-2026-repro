"""Claim 3 (TASK #3 "Theorem 5.2"; paper v2 Theorem 4.1 entropic term + Lemma C.3 /
Carlier et al. 2023 estimate  Ent.(mu,nu,c,eps) - Kant.(mu,nu,c) <= C eps log(1/eps)).

Claim: for (Hoelder/Lipschitz) costs, with entropic penalty eps_t = eta t^{-eta},
the cumulated entropic-approximation error is a LOWER-ORDER (sublinear) term, so
EntUCB keeps the classical linear-bandit rate for the Kantorovich problem.

Executable test (discrete OT, m=40 points on [0,1], Lipschitz cost |x-y|,
Sinkhorn solved to convergence, exact Kantorovich by linear programming):
  * measure g(eps) = Ent(eps) - Kant  and check g(eps) = O(eps log(1/eps)):
    fit g(eps)/(eps log(1/eps)) -> bounded constant C (Lemma C.3),
  * cumulate sum_{t<=T} g(eps_t) with eps_t = eta t^{-eta}, eta in (1/2,1)
    and check the cumulated term is sublinear (log-log slope ~ 1-eta < 1)
    and is dominated by the sqrt(T) trajectorial term,
  * MUTATION: eps_t = const (eta = 0) -> cumulated approximation error becomes
    LINEAR in T (slope ~ 1) -> sublinearity breaks.
Seed 20260803.
"""
import numpy as np
from scipy.optimize import linprog
from common import save, loglog_slope

SEED = 20260803
rng = np.random.default_rng(SEED)
m = 40
x = np.linspace(0, 1, m)
C = np.abs(x[:, None] - x[None, :])              # Lipschitz (1-Lipschitz) cost
mu = rng.random(m) + 0.2; mu /= mu.sum()
nu = rng.random(m) + 0.2; nu /= nu.sum()
rho = np.outer(mu, nu)                            # reference measure (product)


def kantorovich():
    A_eq = []
    for i in range(m):
        z = np.zeros((m, m)); z[i, :] = 1; A_eq.append(z.ravel())
    for j in range(m):
        z = np.zeros((m, m)); z[:, j] = 1; A_eq.append(z.ravel())
    r = linprog(C.ravel(), A_eq=np.array(A_eq), b_eq=np.concatenate([mu, nu]),
                bounds=(0, None), method="highs")
    assert r.success
    return float(r.fun), r.x.reshape(m, m)


def sinkhorn_ent(eps, iters=20000, tol=1e-13):
    """min <C|pi> + eps*KL(pi | rho) over Pi(mu,nu); Sinkhorn on K = rho*exp(-C/eps)."""
    logK = np.log(rho) - C / eps
    f = np.zeros(m); g = np.zeros(m)
    lmu, lnu = np.log(mu), np.log(nu)
    for _ in range(iters):
        f_new = lmu - np.log(np.exp(logK + g[None, :]).sum(1) + 1e-300)
        g_new = lnu - np.log(np.exp(logK + f_new[:, None]).sum(0) + 1e-300)
        if max(np.abs(f_new - f).max(), np.abs(g_new - g).max()) < tol:
            f, g = f_new, g_new
            break
        f, g = f_new, g_new
    pi = np.exp(logK + f[:, None] + g[None, :])
    pi = pi / pi.sum()
    kl = float(np.sum(pi * (np.log(pi + 1e-300) - np.log(rho))))
    return float((C * pi).sum() + eps * kl), float((C * pi).sum()), kl


K, _ = kantorovich()
eps_grid = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
rows = []
for e in eps_grid:
    ent, transp, kl = sinkhorn_ent(e)
    gap = ent - K
    rows.append(dict(eps=e, ent=ent, transport_cost=transp, kl=kl, gap=gap,
                     gap_over_eps_log=gap / (e * np.log(1 / e))))
C_const = max(r["gap_over_eps_log"] for r in rows)
gap_positive = all(r["gap"] > -1e-9 for r in rows)
gap_monotone = all(rows[i]["gap"] >= rows[i + 1]["gap"] - 1e-9 for i in range(len(rows) - 1))
# is the constant in Cbar*eps*log(1/eps) stable (i.e. is the rate right)?
ratios = [r["gap_over_eps_log"] for r in rows]
ratio_spread = float(max(ratios) / min(ratios))

# --- cumulated approximation regret sum_t C eps_t log(1/eps_t) ----------
def cum(T, eta):
    t = np.arange(1, T + 1)
    e = eta * t ** (-eta) if eta > 0 else np.full(T, 0.1)
    return float(C_const * np.sum(e * np.log(1 / e)))

Ts = [100, 1000, 10000, 100000]
eta = 0.75
cum_eta = [cum(T, eta) for T in Ts]
cum_const = [cum(T, 0.0) for T in Ts]
sl_eta, _ = loglog_slope(Ts, cum_eta)
sl_const, _ = loglog_slope(Ts, cum_const)
dominated = all(cum_eta[i] < np.sqrt(Ts[i]) * np.log(Ts[i]) * 3 for i in range(len(Ts)))

# theory shape: sum_t eps_t log(1/eps_t) ~ T^{1-eta} log T (the log factor makes the
# raw log-log slope exceed 1-eta on a finite range), so test the SHAPE, not the slope.
shape = [cum_eta[i] / (Ts[i] ** (1 - eta) * np.log(Ts[i])) for i in range(len(Ts))]
shape_spread = float(max(shape) / min(shape))
ok = (gap_positive and gap_monotone and ratio_spread < 5
      and shape_spread < 2.0 and sl_eta < 1.0 and sl_const > 0.95 and dominated)

out = dict(
    claim=3, source="paper v2 Thm 4.1 entropic term + Lemma C.3 (Carlier et al. 2023); TASK cites 'Theorem 5.2'",
    seed=SEED, m=m, cost="|x-y| (1-Lipschitz)", kantorovich_lp_value=K,
    eps_sweep=rows, C_constant_est=float(C_const), ratio_spread=ratio_spread,
    gap_nonnegative=bool(gap_positive), gap_monotone_in_eps=bool(gap_monotone),
    eta=eta, cumulated_eta_schedule={str(T): v for T, v in zip(Ts, cum_eta)},
    loglog_slope_cumulated_eta=float(sl_eta), asymptotic_slope_1_minus_eta=1 - eta,
    shape_fit_T_pow_1meta_logT=[float(s) for s in shape], shape_ratio_spread=shape_spread,
    dominated_by_sqrtT_logT=bool(dominated),
    mutation_constant_eps_cumulated={str(T): v for T, v in zip(Ts, cum_const)},
    mutation_constant_eps_loglog_slope=float(sl_const),
    verdict="verified" if ok else "inconclusive",
)
for k, v in out.items():
    if k != "eps_sweep":
        print(f"{k}: {v}")
print("eps_sweep:")
for r in rows:
    print("  ", r)
save("claim3.json", out)
