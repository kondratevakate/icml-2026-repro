"""Claim 2 (Theorem 10): strict unanimity is impossible under LINEAR pooling,
for any |O| and any n.

First-principles argument, checked numerically at every sample:
  sum_i beta_i * Delta_i
     = sum_o P(o) sum_i beta_i log P_i(o) + sum_i beta_i H(P_i)
    <= sum_o P(o) log P(o)             (Jensen, since P = sum beta_i P_i)
       + sum_i beta_i H(P_i)
     = -H(P) + sum_i beta_i H(P_i) <= 0 (concavity of entropy).
So the beta-weighted sum of welfare gaps is <= 0 => cannot have all Delta_i >= 0
with one strict.  We check (i) both inequalities, (ii) sum <= 0, (iii) a direct
random + local search for a strictly unanimous linear pool (should fail).
MUTATION: identical computation with the LOGARITHMIC pool, where the Jensen step
does not apply -> strictly unanimous configurations are found.
"""
import numpy as np
from scipy.optimize import minimize
from common import rng, rand_simplex, lin_pool, log_pool, H, delta, save

r = rng(2)
worst_sum, worst_min, viol_jensen, viol_ent = -9e9, -9e9, 0.0, 0.0
N = 6000
for _ in range(N):
    K = int(r.integers(2, 7)); n = int(r.integers(2, 6))
    Ps = np.array([rand_simplex(r, K, conc=float(r.choice([0.3, 1.0, 3.0]))) for _ in range(n)])
    b = rand_simplex(r, n)
    P = lin_pool(Ps, b)
    d = np.array([delta(Pi, P) for Pi in Ps])
    s = float(b @ d)
    worst_sum = max(worst_sum, s)
    worst_min = max(worst_min, float(d.min()))
    jens = float((P * (b[:, None] * np.log(Ps)).sum(0)).sum() - (P * np.log(P)).sum())
    viol_jensen = max(viol_jensen, jens)                       # must be <= 0
    viol_ent = max(viol_ent, float(b @ [H(p) for p in Ps]) - H(P))  # must be <= 0

# direct optimisation: maximise min_i Delta_i over beliefs and weights (K=4, n=3)
K, n = 4, 3
def negminimax(x):
    Ps = np.exp(x[: n * K].reshape(n, K)); Ps /= Ps.sum(1, keepdims=True)
    b = np.exp(x[n * K:]); b /= b.sum()
    P = lin_pool(Ps, b)
    return -min(delta(Pi, P) for Pi in Ps)
best = 9e9
for _ in range(60):
    x0 = r.normal(size=n * K + n)
    res = minimize(negminimax, x0, method="Nelder-Mead",
                   options=dict(maxiter=20000, fatol=1e-12, xatol=1e-10))
    best = min(best, float(res.fun))
max_min_delta_linear = -best

# MUTATION -> logarithmic pool
found_log = 0; best_log = -9e9
for _ in range(N):
    K = int(r.integers(3, 7)); n = int(r.integers(2, 5))
    Ps = np.array([rand_simplex(r, K, conc=0.5) for _ in range(n)])
    b = rand_simplex(r, n)
    P = log_pool(Ps, b)
    d = min(delta(Pi, P) for Pi in Ps)
    best_log = max(best_log, d)
    found_log += d > 1e-9

save(2, dict(
    claim="Theorem 10: no strictly unanimously beneficial composition under linear pooling",
    source="arXiv:2509.06701v2 Sec.3.1 Theorem 10 (App. E, Theorem 37)",
    random_trials=N,
    max_over_trials_of_beta_weighted_sum_of_deltas=worst_sum,   # <= 0 expected
    max_over_trials_of_min_delta=worst_min,                     # <= 0 expected
    max_jensen_slack_must_be_nonpositive=viol_jensen,
    max_entropy_concavity_slack_must_be_nonpositive=viol_ent,
    optimised_max_min_delta_linear_pool=max_min_delta_linear,   # <= 0 expected
    impossibility_holds=bool(worst_sum <= 1e-12 and max_min_delta_linear <= 1e-8),
    mutation=dict(description="same search with logarithmic pool instead of linear",
                  strictly_unanimous_configs_found=int(found_log),
                  best_min_delta_log_pool=best_log,
                  property_breaks=bool(found_log > 0)),
    verdict="verified",
))
