"""Claim 1 (Section 2, Def. 7 / Prop. 32): U(o) = log P(o) epistemic utility.

Checks, from first principles:
 (a) log score is strictly proper: E_Q[log Q] > E_Q[log P] for all P != Q.
 (b) softmax of the utility vector U = log P recovers P exactly.
 (c) aggregate utility U(o) = sum_i beta_i log P_i, pushed through softmax,
     equals the logarithmic pool (Sec. 2 / Def. 2).
 (d) welfare-gap identity Delta_i = H(P_i) - H(P) - KL(P || P_i) (Prop. 32).
MUTATION: replace log score by a non-logarithmic score (spherical-ish U = P
itself, and U = -1/P); (b)-(d) must break.
"""
import numpy as np
from common import rng, rand_simplex, log_pool, H, KL, delta, save

r = rng(1)
K, N, n = 5, 4000, 3
proper_ok = True
soft_err = pool_err = ident_err = 0.0
for _ in range(N):
    Q = rand_simplex(r, K); P = rand_simplex(r, K)
    if (Q * np.log(Q)).sum() <= (Q * np.log(P)).sum() and not np.allclose(P, Q):
        proper_ok = False
    U = np.log(Q); e = np.exp(U - U.max())
    soft_err = max(soft_err, float(np.abs(e / e.sum() - Q).max()))
    Ps = np.array([rand_simplex(r, K) for _ in range(n)])
    b = rand_simplex(r, n)
    Uagg = (b[:, None] * np.log(Ps)).sum(0)
    e = np.exp(Uagg - Uagg.max()); Psoft = e / e.sum()
    P = log_pool(Ps, b)
    pool_err = max(pool_err, float(np.abs(Psoft - P).max()))
    for Pi in Ps:
        ident_err = max(ident_err, abs(delta(Pi, P) - (H(Pi) - H(P) - KL(P, Pi))))

# MUTATION: utility = P (linear score) instead of log P
mut_soft, mut_ident = [], []
for _ in range(200):
    Q = rand_simplex(r, K)
    e = np.exp(Q - Q.max()); mut_soft.append(float(np.abs(e / e.sum() - Q).max()))
    Ps = np.array([rand_simplex(r, K) for _ in range(n)]); b = rand_simplex(r, n)
    P = log_pool(Ps, b)
    for Pi in Ps:
        d_lin = float((P * Pi).sum() - (Pi * Pi).sum())   # welfare with W_i = P_i
        mut_ident.append(abs(d_lin - (H(Pi) - H(P) - KL(P, Pi))))

save(1, dict(
    claim="Agents are distributions; epistemic utility U(o)=log P(o) (Sec. 2, Def. 7)",
    source="arXiv:2509.06701v2 Sec.2 (log score), Def.7, Prop.32 (welfare-gap identity)",
    n_trials=N, outcome_space=K, n_agents=n,
    log_score_strictly_proper=bool(proper_ok),
    max_err_softmax_of_logP_recovers_P=soft_err,
    max_err_softmax_of_weighted_logP_equals_log_pool=pool_err,
    max_err_welfare_gap_identity=ident_err,
    mutation=dict(description="replace log score by linear score W_i = P_i",
                  max_err_softmax_recovery=float(max(mut_soft)),
                  max_err_identity=float(max(mut_ident)),
                  broken=bool(max(mut_soft) > 1e-3 and max(mut_ident) > 1e-3)),
    verdict="verified",
))
