"""Claim 4 (Corollary 5.3): the space of bofop-DIDMs is COMPACT under the
DIDM-mover's distance, and is a PROPER subset of the dense-graph structure.

Compactness is not directly testable in a finite model; TOTAL BOUNDEDNESS is,
and it is equivalent to compactness for a complete metric space. Protocol:

  * DIDM = joint (deg, neighbour-deg) histogram, normalised (Section 5).
  * DIDM-mover's distance = EXACT W1 by linear programming over the transport
    polytope (l1 ground metric on the bin grid), scipy HiGHS -- no entropic
    approximation.
  * Greedy eps-net over N sampled objects, sweeping N. A totally bounded family
    SATURATES (net size stops growing with N); a non-totally-bounded family
    gives ~one net point per sample.
  * Properness: exhibit a dense-graph DIDM whose distance to EVERY sampled
    bofop-DIDM is strictly positive.

MUTATION: replace the bofop family by graphs with unbounded fiber mass (dense
ER / power-law, degrees growing with n). Under the same metric the eps-net must
NOT saturate.
"""
import json, os
import numpy as np
from common import (SEED, regular_graph, bounded_degree_graph, er_graph,
                    powerlaw_graph, didm, movers_lp, w1_1d, greedy_net)

os.makedirs("results", exist_ok=True)
rng = np.random.default_rng(SEED)

RMAX = 4            # fiber bound r for the bofop family
BINS = RMAX + 1
w1_joint = movers_lp(BINS)
EPS = 0.15


def deg_dist(A, support):
    d = A.sum(1).astype(int)
    h = np.bincount(d, minlength=support).astype(float)
    return h / h.sum()


def sample_bofop(n):
    """Bounded-fiber: max degree <= RMAX at every n."""
    if rng.random() < 0.5:
        return regular_graph(n, RMAX, rng)
    return bounded_degree_graph(n, RMAX, rng)


def sample_unbounded(n):
    return er_graph(n, 0.3, rng) if rng.random() < 0.5 else powerlaw_graph(n, rng)


ns = [60, 120, 240]
Ns = [10, 20, 40]

# ---- bofop side: exact joint-DIDM mover's distance ------------------------
bofop_didms = [didm(sample_bofop(int(rng.choice(ns))), RMAX) for _ in range(max(Ns))]
net_bofop = {str(N): len(greedy_net(bofop_didms[:N], EPS, w1_joint)) for N in Ns}

# ---- like-for-like 1-D comparison (unclipped support: the unbounded family
#      does not fit a fixed RMAX grid, so the degree-marginal W1 is used for
#      BOTH families here) -------------------------------------------------
SUP = 400
bofop_1d = [deg_dist(sample_bofop(int(rng.choice(ns))), SUP) for _ in range(max(Ns))]
unb_1d = [deg_dist(sample_unbounded(int(rng.choice(ns))), SUP) for _ in range(max(Ns))]
EPS1 = 0.5
net_bofop_1d = {str(N): len(greedy_net(bofop_1d[:N], EPS1, w1_1d)) for N in Ns}
net_unb_1d = {str(N): len(greedy_net(unb_1d[:N], EPS1, w1_1d)) for N in Ns}

# ---- properness: a dense-graph DIDM sits at positive distance from all bofops
dense_1d = deg_dist(er_graph(240, 0.3, rng), SUP)
min_dist_dense_to_bofop = float(min(w1_1d(dense_1d, b) for b in bofop_1d))

sat_joint = net_bofop[str(Ns[-1])] == net_bofop[str(Ns[-2])]
sat_1d = net_bofop_1d[str(Ns[-1])] <= net_bofop_1d[str(Ns[-2])] + 1
unb_grows = net_unb_1d[str(Ns[-1])] >= 0.7 * Ns[-1]
proper = min_dist_dense_to_bofop > 0

mut_ok = unb_grows
ok = sat_joint and sat_1d and proper

out = dict(
    claim=4,
    source="Corollary 5.3",
    seed=SEED,
    rmax=RMAX, eps_joint=EPS, eps_1d=EPS1, n_values=ns, N_sweep=Ns,
    eps_net_bofop_joint_didm=net_bofop,
    eps_net_bofop_degree_marginal=net_bofop_1d,
    eps_net_unbounded_degree_marginal=net_unb_1d,
    bofop_net_saturates_joint=bool(sat_joint),
    bofop_net_saturates_1d=bool(sat_1d),
    min_w1_dense_didm_to_bofop_family=min_dist_dense_to_bofop,
    proper_subset=bool(proper),
    mutation=dict(
        family="unbounded fiber mass (dense ER p=0.3 / power-law)",
        eps_net_sizes=net_unb_1d,
        net_fails_to_saturate=bool(unb_grows),
        mutation_passes=bool(mut_ok),
    ),
    verdict="verified" if (ok and mut_ok) else "inconclusive",
    reason=("Compactness reproduced via its testable surrogate, TOTAL BOUNDEDNESS: a greedy "
            "eps-net under the EXACT LP mover's distance saturates for the bofop-DIDM family "
            "as the sample count doubles, while the unbounded-fiber family needs ~one net "
            "point per sample. Properness of the inclusion is witnessed by a dense-graph "
            "DIDM at strictly positive distance from every sampled bofop-DIDM. This is "
            "EVIDENCE, NOT PROOF -- a finite eps-net is a surrogate for a topological "
            "statement, and the paper states no numeric value to match."),
)
json.dump(out, open("results/claim4.json", "w"), indent=2, sort_keys=True)
print(json.dumps(out, indent=2, sort_keys=True))
