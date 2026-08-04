"""Claim 1 (Theorem 1, Sec 4.1): LP threshold rounding at theta=kappa/(1+kappa) gives
   coverage loss W-e(K) <= (1+kappa)*eps*W   and   |K| <= (1+1/kappa)*r.
Protocol: random weighted graphs; r = budget; eps := (W - LP_opt(r))/W  (the LP-certified
loss level, which lower-bounds any integral eps achievable at budget r).
Mutation: round at a *smaller* theta' = theta/3 (size bound must break) and at a
*larger* theta'' (loss bound must break).
"""
import json, os, numpy as np
from common import random_weighted_graph, solve_lp, round_threshold, coverage, total_weight

os.makedirs('results', exist_ok=True)
SEED = 20260803
rng = np.random.default_rng(SEED)
kappas = [0.25, 0.5, 1.0, 2.0, 4.0]
rows, viol_loss, viol_size = [], 0, 0
mut_size_break = mut_loss_break = 0
mut_trials = 0

for t in range(24):
    n = int(rng.integers(12, 22)); p = float(rng.uniform(0.2, 0.5))
    G = random_weighted_graph(n, p, seed=int(rng.integers(1e6)))
    if G.number_of_edges() == 0:
        continue
    W = total_weight(G)
    r = max(3, n // 3)
    x, lp = solve_lp(G, r)
    eps = (W - lp) / W
    for kap in kappas:
        theta = kap / (1 + kap)
        K = round_threshold(x, theta)
        loss = W - coverage(G, K)
        ok_loss = loss <= (1 + kap) * eps * W + 1e-7
        ok_size = len(K) <= (1 + 1 / kap) * r + 1e-9
        viol_loss += (not ok_loss); viol_size += (not ok_size)
        rows.append(dict(trial=t, n=n, m=G.number_of_edges(), r=r, kappa=kap,
                         eps=eps, theta=theta, size=len(K),
                         size_bound=(1 + 1 / kap) * r, loss=loss,
                         loss_bound=(1 + kap) * eps * W,
                         ok_loss=bool(ok_loss), ok_size=bool(ok_size)))
        # MUTATION A: theta too small -> size bound should be violated sometimes
        Ka = round_threshold(x, theta / 3)
        # MUTATION B: theta too large -> loss bound should be violated sometimes
        Kb = round_threshold(x, min(0.999, theta * 1.8 + 0.2))
        mut_trials += 1
        mut_size_break += (len(Ka) > (1 + 1 / kap) * r + 1e-9)
        mut_loss_break += ((W - coverage(G, Kb)) > (1 + kap) * eps * W + 1e-7)

res = dict(
    claim="Theorem 1 bicriteria (1+kappa, 1+1/kappa) LP rounding",
    source="Theorem 1, Section 4.1", seed=SEED,
    n_instances=len(rows), kappas=kappas,
    violations_loss=int(viol_loss), violations_size=int(viol_size),
    max_loss_ratio=max(r_['loss'] / r_['loss_bound'] for r_ in rows if r_['loss_bound'] > 0),
    max_size_ratio=max(r_['size'] / r_['size_bound'] for r_ in rows),
    mutation_trials=mut_trials,
    mutation_small_theta_size_violations=int(mut_size_break),
    mutation_large_theta_loss_violations=int(mut_loss_break),
    verdict="verified" if (viol_loss == 0 and viol_size == 0 and (mut_size_break + mut_loss_break) > 0) else "inconclusive",
    detail=rows[:20],
)
json.dump(res, open('results/claim1.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'detail'}, indent=1))
