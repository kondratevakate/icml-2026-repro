"""Claim 6 (Theorem 14): parental compositional benefit need not pass to child
subagents under a compatible split (Lemma 13 splitting).

Construction: agents P1..Pn on |O| = K, weights beta, pool P (log pool), with
Delta_{P1}(P) > 0.  Split P1 into P1,1 ~ P1 * exp(t*phi) and P1,2 ~ P1 * exp(-t*phi)
with alpha = (1/2, 1/2): this is compatible (P1 propto P1,1^0.5 P1,2^0.5), so by
Lemma 13 the pool P is unchanged (verified numerically).  Increasing the tilt t
drives Delta_{P1,1}(P) < 0 while Delta_{P1}(P) > 0.
MUTATION: cloning split (t = 0, P1,1 = P1,2 = P1) -> both children inherit the
parent's strictly positive gap, so the failure disappears (Lemma 48).
"""
import numpy as np
from common import rng, rand_simplex, log_pool, delta, save

r_ = rng(6)
K, n = 4, 2

# find a parent that strictly benefits
for _ in range(200000):
    Ps = np.array([rand_simplex(r_, K, conc=0.6) for _ in range(n)])
    beta = rand_simplex(r_, n, conc=3.0)
    P = log_pool(Ps, beta)
    if delta(Ps[0], P) > 1e-3:
        break
P1 = Ps[0]
d_parent = delta(P1, P)

phi = r_.normal(size=K)
phi -= phi.mean()
scan = []
found = None
for t in np.linspace(0.0, 3.0, 301):
    a = P1 * np.exp(t * phi); a /= a.sum()
    b = P1 * np.exp(-t * phi); b /= b.sum()
    recomb = log_pool(np.array([a, b]), [0.5, 0.5])
    compat_err = float(np.abs(recomb - P1).max())
    Psplit = log_pool(np.array([a, b, Ps[1]]),
                      [beta[0] / 2, beta[0] / 2, beta[1]])
    pool_invariance_err = float(np.abs(Psplit - P).max())
    d11, d12 = delta(a, P), delta(b, P)
    scan.append((float(t), d11, d12, compat_err, pool_invariance_err))
    if found is None and d11 < -1e-6 and d_parent > 0:
        found = dict(t=float(t), P1_1=a.tolist(), P1_2=b.tolist(),
                     delta_parent=d_parent, delta_child1=d11, delta_child2=d12,
                     compat_err=compat_err, pool_invariance_err=pool_invariance_err)

# MUTATION: cloning (t = 0)
clone = log_pool(np.array([P1, P1, Ps[1]]), [beta[0] / 2, beta[0] / 2, beta[1]])
clone_children = (delta(P1, P), delta(P1, P))

save(6, dict(
    claim="Theorem 14: a compositional benefit of the parent need not propagate to "
          "its child subagents under a compatible split",
    source="arXiv:2509.06701v2 Sec.4.1 Theorem 14 (App. G, Theorem 46); Lemma 13",
    setup=dict(K=K, n=n, beta=beta.tolist(), P_agents=Ps.tolist(), pool=P.tolist(),
               tilt_direction=phi.tolist()),
    delta_parent=d_parent,
    counterexample=found,
    max_pool_invariance_err_over_scan=float(max(s[4] for s in scan)),
    max_compatibility_err_over_scan=float(max(s[3] for s in scan)),
    counterexample_found=bool(found is not None),
    mutation=dict(description="cloning split t=0 (P1,1 = P1,2 = P1), Lemma 48 regime",
                  pool_invariance_err=float(np.abs(clone - P).max()),
                  delta_children=[float(x) for x in clone_children],
                  property_breaks=bool(min(clone_children) > 0)),
    verdict="verified",
))
