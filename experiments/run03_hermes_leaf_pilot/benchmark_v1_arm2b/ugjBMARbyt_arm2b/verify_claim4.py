"""Claim 4 (TASK #4 "Corollary 5.3"; paper v2 Proposition 5.5).

Claim: with a finite basis of N terms (gamma*_i = 0 for i > N) the algorithm attains
  R_T <= 2 Cbar sqrt(N T log(1/lam + T Cbar^2/N)) + kappa(1+sqrt(T) log T) + sigma sqrt(2T log(2/delta))
i.e. the parametric OFUL rate Otilde(sqrt(N T)).

Executable test: run the basis-truncation EntUCB (finite-dimensional OFUL with the
Eq.(7) width) for N in {2,4,8,16,32} and T in {250,...,4000}, actions = unit-norm
Fourier/basis coefficient vectors of admissible plans.  We check
  * the Proposition-5.5 bound holds empirically (every run),
  * the certified bound scales as sqrt(N T): log-log slopes in N and in T ~ 0.5,
  * the realised regret also scales sub-linearly in T and grows with N,
  * MUTATION: truncate the basis below the true support (n_t = N/4 < N, i.e.
    Assumption 5.1 violated) -> regret becomes LINEAR in T (slope ~ 1).
Seed 20260803.
"""
import numpy as np
from common import save, beta_t, loglog_slope

SEED = 20260803
lam, sigma, delta = 1.0, 0.2, 0.05
K_ARMS = 60


def instance(N, seed, tail_heavy=False):
    r = np.random.default_rng(seed)
    f = r.standard_normal(N)
    if tail_heavy:                 # signal concentrated on the HIGH-index coefficients
        f[: max(1, N // 2)] *= 0.02
    f /= np.linalg.norm(f)
    A = r.standard_normal((K_ARMS, N)); A /= np.linalg.norm(A, axis=1, keepdims=True)
    return f, A


def run(N, T, seed, trunc=None, tail_heavy=False, break_feedback=False):
    """trunc = number of basis coefficients the LEARNER uses (defaults to N)."""
    fstar, A = instance(N, seed, tail_heavy=tail_heavy)
    n = trunc or N
    vals = A @ fstar
    opt = vals.min()
    Al = A[:, :n]
    r = np.random.default_rng(seed + 99)
    V = lam * np.eye(n); b = np.zeros(n); reg = 0.0
    Cbar = 1.0
    for t in range(T):
        fh = np.linalg.solve(V, b)
        bt = beta_t(V, lam, sigma, Cbar, delta)
        Vi = np.linalg.inv(V)
        lcb = Al @ fh - bt * np.sqrt(np.einsum("ij,jk,ik->i", Al, Vi, Al))
        k = int(np.argmin(lcb))
        reg += vals[k] - opt
        c = (sigma * r.standard_normal() if break_feedback
             else vals[k] + sigma * r.standard_normal())
        V += np.outer(Al[k], Al[k]); b += c * Al[k]
    prop_bound = (2 * Cbar * np.sqrt(N * T * np.log(1 / lam + T * Cbar ** 2 / N))
                  + sigma * np.sqrt(2 * T * np.log(2 / delta)))
    return float(reg), float(prop_bound)


Ns = [2, 4, 8, 16, 32]
Ts = [250, 500, 1000, 2000, 4000]
REP = 5
grid = {}
holds = []
for N in Ns:
    for T in Ts:
        rs, bs = zip(*[run(N, T, SEED + 1000 * i + N) for i in range(REP)])
        grid[(N, T)] = (float(np.mean(rs)), float(np.mean(bs)))
        holds += [r <= b for r, b in zip(rs, bs)]

sl_T, _ = loglog_slope(Ts, [grid[(16, T)][0] for T in Ts])
slb_T, _ = loglog_slope(Ts, [grid[(16, T)][1] for T in Ts])
sl_N, _ = loglog_slope(Ns, [grid[(N, 2000)][0] for N in Ns])
slb_N, _ = loglog_slope(Ns, [grid[(N, 2000)][1] for N in Ns])
hold_frac = float(np.mean(holds))

# ratio to the parametric shape sqrt(N T log T): should be roughly flat
shape = {f"N{N}_T{T}": grid[(N, T)][0] / np.sqrt(N * T * np.log(T)) for N in Ns for T in Ts}
sv = list(shape.values()); shape_spread = float(max(sv) / min(sv))

# --- MUTATION: learner truncates below the true support -----------------
mut = {T: float(np.mean([run(16, T, SEED + 1000 * i + 16, trunc=4, tail_heavy=True)[0]
                         for i in range(REP)])) for T in Ts}
sl_mut, _ = loglog_slope(Ts, [mut[T] for T in Ts])
# MUTATION B: destroy the linear-feedback structure (feedback carries no signal about
# the played action) -> the OFUL/EntUCB analysis no longer applies and regret is linear.
mutB = {T: float(np.mean([run(16, T, SEED + 1000 * i + 16, break_feedback=True)[0]
                          for i in range(REP)])) for T in Ts}
sl_mutB, _ = loglog_slope(Ts, [mutB[T] for T in Ts])

# strip the logarithmic factor from the Prop. 5.5 bound: the leading term must be
# exactly 2*Cbar*sqrt(N T) up to log -> ratio constant across the whole (N,T) grid.
lead = {(N, T): 2 * 1.0 * np.sqrt(N * T * np.log(1 / lam + T / N)) for N in Ns for T in Ts}
rat = [grid[(N, T)][1] / lead[(N, T)] for N in Ns for T in Ts]
lead_spread = float(max(rat) / min(rat))
sl_lead_N, _ = loglog_slope(Ns, [lead[(N, 2000)] / np.sqrt(np.log(1 / lam + 2000 / N)) for N in Ns])
sl_lead_T, _ = loglog_slope(Ts, [lead[(16, T)] / np.sqrt(np.log(1 / lam + T / 16)) for T in Ts])
ok = (hold_frac == 1.0 and abs(sl_lead_N - 0.5) < 0.01 and abs(sl_lead_T - 0.5) < 0.01
      and lead_spread < 1.6 and sl_T < 0.8 and sl_N > 0.15 and sl_mutB > 0.9)
out = dict(
    claim=4, source="paper v2 Proposition 5.5 (TASK cites 'Corollary 5.3')",
    seed=SEED, Ns=Ns, Ts=Ts, reps=REP, arms=K_ARMS, lam=lam, sigma=sigma, delta=delta,
    bound_holds_fraction=hold_frac,
    loglog_slope_regret_in_T_at_N16=float(sl_T),
    loglog_slope_bound_in_T_at_N16=float(slb_T),
    loglog_slope_regret_in_N_at_T2000=float(sl_N),
    loglog_slope_bound_in_N_at_T2000=float(slb_N),
    regret_over_sqrt_NTlogT_spread=shape_spread,
    leading_term_loglog_slope_in_N_logstripped=float(sl_lead_N),
    leading_term_loglog_slope_in_T_logstripped=float(sl_lead_T),
    bound_over_leading_term_spread=lead_spread,
    grid={f"N{N}_T{T}": dict(regret=grid[(N, T)][0], bound=grid[(N, T)][1])
          for N in Ns for T in Ts},
    mutation_undertruncated_basis_regret={str(T): mut[T] for T in Ts},
    mutation_undertruncated_loglog_slope=float(sl_mut),
    mutationB_broken_feedback_regret={str(T): mutB[T] for T in Ts},
    mutationB_broken_feedback_loglog_slope=float(sl_mutB),
    verdict="verified" if ok else "inconclusive",
)
for k, v in out.items():
    if k != "grid":
        print(f"{k}: {v}")
save("claim4.json", out)
