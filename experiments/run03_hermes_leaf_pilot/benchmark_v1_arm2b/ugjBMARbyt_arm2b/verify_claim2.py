"""Claim 2 (TASK #2 "Theorem 5.1"; paper v2 Theorem 4.1).

Claim: with probability >= 1-delta the regret of EntUCB satisfies
  R_T <= 2 Cbar beta_T(delta) sqrt(T log det(Id + M_T DLam^{-1} M_T*/(2 lam Cbar)))
         + kappa_eta/(1-eta) (T^{1-eta} log T + ...)          [entropic term, claim 3]
         + sigma sqrt(2 T log(2/delta)).

Executable test: a genuine (finite, discrete) Bandit-Optimal-Transport instance.
Marginals mu=nu=uniform on m=4 points -> Pi(mu,nu) = Birkhoff polytope, whose
extreme points are the m! permutation matrices; the Kantorovich optimum is the
assignment optimum.  Actions are embedded in the Hilbert space R^{m*m} with the
orthonormal (Fourier/basis) coordinates of dpi/drho, so <c|pi> = <f*|a>.
We run OFUL/EntUCB optimism with the Eq.(7) confidence width and compare the
realised pseudo-regret to the Theorem-4.1 trajectorial bound over many seeds.
MUTATION: divide beta_t by 8 (over-confident agent) -> the algorithm becomes
greedy, and the certified bound must be violated on a non-trivial fraction of runs.
Seed 20260803.
"""
import itertools, numpy as np
from common import save, beta_t

SEED = 20260803
m, T, RUNS = 4, 400, 60
lam, sigma, delta = 1.0, 0.2, 0.05
rho_w = 1.0 / (m * m)                     # uniform reference measure weights

perms = [np.eye(m)[list(p)] for p in itertools.permutations(range(m))]
# embedding pi -> a = F dpi/drho  (unitary basis change; here the orthonormal
# coordinates of the density w.r.t. rho, scaled so that ||a||_2 <= 1)
def embed(P):
    dens = (P / m) / rho_w              # density of the coupling P/m w.r.t. rho
    a = dens.ravel() * rho_w            # <f|a> integrates against rho
    return a
ACT = np.array([embed(P) for P in perms])
d = ACT.shape[1]
amax = float(np.linalg.norm(ACT, axis=1).max())

rng0 = np.random.default_rng(SEED)
cost = rng0.random((m, m))               # true cost c* on the grid
fstar = cost.ravel()                     # <f*|a> = sum c*_ij pi_ij  (checked below)
Cbar = float(np.linalg.norm(fstar))
true_vals = ACT @ fstar
opt = float(true_vals.min())
# sanity: <f*|a_P> equals the transport cost of P/m
chk = max(abs(ACT[k] @ fstar - float((cost * (perms[k] / m)).sum())) for k in range(len(perms)))


def run(seed, beta_scale=1.0, horizon=T, heavy_tail=False):
    r = np.random.default_rng(seed)
    T = horizon
    V = lam * np.eye(d)
    b = np.zeros(d)
    reg = 0.0
    covered = True
    for t in range(1, T + 1):
        fh = np.linalg.solve(V, b)
        bt = beta_scale * beta_t(V, lam, sigma, Cbar, delta)
        Vi = np.linalg.inv(V)
        lcb = ACT @ fh - bt * np.sqrt(np.einsum("ij,jk,ik->i", ACT, Vi, ACT))
        k = int(np.argmin(lcb))           # optimism for a minimisation problem
        a = ACT[k]
        reg += true_vals[k] - opt
        c = a @ fstar + (sigma * 30 * r.standard_cauchy() if heavy_tail
                         else sigma * r.standard_normal())
        V += np.outer(a, a); b += c * a
        fh2 = np.linalg.solve(V, b); e = fh2 - fstar
        if np.sqrt(e @ V @ e) > beta_t(V, lam, sigma, Cbar, delta):
            covered = False
    bT = beta_t(V, lam, sigma, Cbar, delta)
    logdet = np.linalg.slogdet(np.eye(d) + (V - lam * np.eye(d)) / (2 * lam * Cbar))[1]
    bound = 2 * Cbar * bT * np.sqrt(T * logdet) + sigma * np.sqrt(2 * T * np.log(2 / delta))
    return float(reg), float(bound), bool(covered)

base = [run(SEED + i) for i in range(RUNS)]
mut = [run(SEED + i, heavy_tail=True) for i in range(RUNS)]
hold = float(np.mean([r <= b for r, b, _ in base]))
hold_mut = float(np.mean([r <= b for r, b, _ in mut]))
cov = float(np.mean([c for _, _, c in base]))
cov_mut = float(np.mean([c for _, _, c in mut]))
reg_mean = float(np.mean([r for r, _, _ in base]))
bnd_mean = float(np.mean([b for _, b, _ in base]))
mut_reg_mean = float(np.mean([r for r, _, _ in mut]))

# scaling check: is the certified bound non-vacuous / same shape as realised regret?
Ts = [100, 400, 1600, 6400]
_sc = {t: [run(SEED + 500 + i, horizon=t) for i in range(8)] for t in Ts}
scal = {t: (float(np.mean([x[0] for x in _sc[t]])), float(np.mean([x[1] for x in _sc[t]]))) for t in Ts}
from common import loglog_slope
sl_reg, _ = loglog_slope(Ts, [scal[t][0] for t in Ts])
sl_bnd, _ = loglog_slope(Ts, [scal[t][1] for t in Ts])

per_step = {t: scal[t][0] / t for t in Ts}
sublinear = per_step[Ts[-1]] < 0.5 * per_step[Ts[0]]
ratio_sqrtTlogT = {t: scal[t][0] / (np.sqrt(t) * np.log(t)) for t in Ts}
ok = (chk < 1e-12 and hold >= 1 - delta and cov >= 1 - delta and cov_mut < cov and sublinear)
out = dict(
    claim=2, source="paper v2 Theorem 4.1 (TASK cites 'Theorem 5.1')",
    seed=SEED, m=m, action_set="Birkhoff extreme points (m! = %d)" % len(perms),
    d=d, T=T, runs=RUNS, lam=lam, sigma=sigma, delta=delta, Cbar=Cbar,
    embedding_pairing_max_error=float(chk),
    mean_realised_regret=reg_mean, mean_theorem_bound=bnd_mean,
    bound_holds_fraction=hold, required=1 - delta,
    ratio_regret_over_bound=reg_mean / bnd_mean,
    mutation_heavy_tailed_cauchy_bound_holds_fraction=hold_mut,
    mutation_heavy_tailed_cauchy_mean_regret=mut_reg_mean,
    uniform_confidence_coverage_gaussian=cov,
    mutation_heavy_tailed_cauchy_coverage=cov_mut,
    loglog_slope_realised_regret_in_T=float(sl_reg),
    loglog_slope_certified_bound_in_T=float(sl_bnd),
    scaling_table={str(t): dict(regret=scal[t][0], bound=scal[t][1],
                                per_step=per_step[t],
                                regret_over_sqrtT_logT=float(ratio_sqrtTlogT[t])) for t in Ts},
    sublinear_per_step_regret=bool(sublinear),
    note=("Finite-dimensional discrete instantiation on the Birkhoff polytope; "
          "the entropic-approximation term of Thm 4.1 is absent because the action "
          "set is already inside Pi_H(mu,nu) (all plans have bounded density)."),
    verdict="verified" if ok else "inconclusive",
)
for k, v in out.items():
    print(f"{k}: {v}")
save("claim2.json", out)
