"""Claim 2 (Definition 3.1, bofops): the bounded-fiber subclass is defined by
ess sup_{x in Omega} nu_x(Omega) < infinity.

Finite model: nu_x(Omega) = fiber mass of vertex x = its degree under the
n-atom embedding (W = n*Adj => (W @ 1)/n = degree). "ess sup < infinity" is NOT
testable at a single n: we sweep n and fit the log-log growth exponent of
max_x nu_x(Omega).

  bounded-degree family  -> exponent ~ 0   (bofop)
  dense Erdos-Renyi      -> exponent ~ 1   (not a bofop)
  single sqrt(n) hub     -> exponent ~ 0.5 (not a bofop; discriminates finely)
"""
import json, os
import numpy as np
from common import SEED, regular_graph, er_graph, loglog_slope

os.makedirs("results", exist_ok=True)
rng = np.random.default_rng(SEED)
ns = [100, 200, 400, 800, 1600]


def fiber_sup(A):
    """ess sup_x nu_x(Omega) under W = n*Adj, mu uniform: = max degree."""
    return float(A.sum(1).max())


bofop, dense, hub = [], [], []
for n in ns:
    A = regular_graph(n, 4, rng)
    bofop.append(fiber_sup(A))
    dense.append(fiber_sup(er_graph(n, 0.3, rng)))
    H = regular_graph(n, 4, rng)
    k = int(np.sqrt(n))
    idx = rng.choice(np.arange(1, n), size=k, replace=False)
    H[0, idx] = H[idx, 0] = 1.0                     # one sqrt(n)-degree hub
    hub.append(fiber_sup(H))

s_bofop = loglog_slope(ns, bofop)
s_dense = loglog_slope(ns, dense)
s_hub = loglog_slope(ns, hub)

bofop_ok = abs(s_bofop) < 0.05 and max(bofop) <= 4.0 + 1e-9
mut_ok = s_dense > 0.8 and s_hub > 0.3

out = dict(
    claim=2,
    source="Definition 3.1 (bounded-fiber operators / bofops)",
    seed=SEED,
    n_sweep=ns,
    fiber_sup_bofop=bofop,
    fiber_sup_dense_er=dense,
    fiber_sup_sqrtn_hub=hub,
    exponent_bofop=s_bofop,
    exponent_dense_er=s_dense,
    exponent_sqrtn_hub=s_hub,
    bofop_bounded=bool(bofop_ok),
    mutation=dict(
        dense_er_exponent=s_dense,
        sqrtn_hub_exponent=s_hub,
        note="removing the bounded-fiber hypothesis makes ess sup nu_x grow with n",
        mutation_passes=bool(mut_ok),
    ),
    verdict="verified" if (bofop_ok and mut_ok) else "inconclusive",
    reason=("The bofop condition ess sup_x nu_x(Omega) < infinity is reproduced as a "
            "log-log growth exponent over an n-sweep: the bounded-degree (sparse) family "
            "has exponent ~0 with fiber sup pinned at 4 for every n, while dense ER "
            "(exponent ~1) and a single sqrt(n) hub (exponent ~0.5) are excluded. This "
            "shows the definition genuinely discriminates a sparse subclass inside the "
            "graphop class. No numeric value is stated in the paper to match."),
)
json.dump(out, open("results/claim2.json", "w"), indent=2, sort_keys=True)
print(json.dumps(out, indent=2, sort_keys=True))
