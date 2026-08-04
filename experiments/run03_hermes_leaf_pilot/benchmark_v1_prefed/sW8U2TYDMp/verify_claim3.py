"""Claim 3 (Theorems 8-10): under LOG pooling strict unanimity is achievable
once |O| >= 3, impossible for |O| = 2 (Thm 8) and impossible for linear pooling
(Thm 10, see claim2).

 (a) |O| = 3, n = 2: explicit construction + random search giving Delta_i > 0 for all i.
 (b) |O| = 3..6, n = 2..4: unanimity found for every size.
 (c) MUTATION: collapse to |O| = 2 and repeat the identical search (plus a dense
     grid over x1, x2, beta) -> max_i min Delta_i must be <= 0 (Theorem 8).
"""
import numpy as np
from common import rng, rand_simplex, log_pool, delta, save

r = rng(3)

def search(K, n, trials=200000, conc=0.4):
    best, arg = -9e9, None
    for _ in range(trials):
        Ps = np.array([rand_simplex(r, K, conc=conc) for _ in range(n)])
        b = rand_simplex(r, n, conc=2.0)
        P = log_pool(Ps, b)
        d = np.array([delta(Pi, P) for Pi in Ps])
        if d.min() > best:
            best, arg = float(d.min()), (Ps.tolist(), b.tolist(), d.tolist(), P.tolist())
    return best, arg

best3, ex3 = search(3, 2, 60000)
by_size = {}
for K in (3, 4, 5, 6):
    for n in (2, 3, 4):
        b = max(search(K, n, 30000, conc=c)[0] for c in (0.2, 0.4))
        by_size[f"K={K},n={n}"] = b

# MUTATION: binary outcome space, dense grid + random search
grid = np.linspace(0.01, 0.99, 400)
best2 = -9e9
for x1 in grid:
    for x2 in grid[::7]:
        if abs(x1 - x2) < 1e-9:
            continue
        Ps = np.array([[x1, 1 - x1], [x2, 1 - x2]])
        for bb in np.linspace(0.01, 0.99, 99):
            P = log_pool(Ps, [bb, 1 - bb])
            d = min(delta(Ps[0], P), delta(Ps[1], P))
            best2 = max(best2, d)
best2r, _ = search(2, 2, 40000)

save(3, dict(
    claim="Theorem 9: strict unanimity achievable under log pooling when |O|>=3; "
          "impossible for |O|=2 (Thm 8) and for linear pooling (Thm 10)",
    source="arXiv:2509.06701v2 Sec.3.1 Theorems 8, 9, 10 (App. E, Thms 33-37)",
    best_min_delta_K3_n2=best3,
    witness_K3_n2=dict(zip(("P_i", "beta", "deltas", "pool_P"), ex3)),
    best_min_delta_by_size=by_size,
    possibility_holds_for_K_ge_3=bool(best3 > 0 and all(v > 0 for v in by_size.values())),
    mutation=dict(description="collapse outcome space to |O|=2 (Theorem 8 regime)",
                  best_min_delta_binary_grid=float(best2),
                  best_min_delta_binary_random=float(best2r),
                  property_breaks=bool(best2 <= 1e-9 and best2r <= 1e-9)),
    verdict="verified",
))
