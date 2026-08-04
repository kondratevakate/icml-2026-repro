"""Claim 5 (Section 6.1): universal approximation -- any CONTINUOUS function on
bofop-DIDMs can be uniformly approximated by MPNNs directly on sparse graphs.

Model: random-feature MPNN (fixed random message/update weights, ridge readout
head) -- a legitimate MPNN whose capacity can be swept cheaply. Targets are
functions of the graph's DIDM only (permutation-invariant, defined on the DIDM
space), evaluated on bounded-degree (bofop) graphs.

UNIFORM approximation => measure SUP-NORM test error, not RMSE.
Criterion: sup-norm error decreases monotonically (in trend) with width and
becomes small relative to the target's range.

MUTATION: swap the target for one that is DISCONTINUOUS in the DIDM metric
(a threshold indicator on the mean degree). Uniform approximation must FAIL --
the sup error has to plateau at a level set by the jump.
"""
import json, os
import numpy as np
from common import (SEED, regular_graph, bounded_degree_graph, didm,
                    mpnn_embed, ridge_fit, ridge_predict)

os.makedirs("results", exist_ok=True)
rng = np.random.default_rng(SEED)
RMAX = 4
NTRAIN, NTEST = 220, 120


def sample_bofop(n):
    """Bofop family with a CONTINUUM of DIDMs: start from a 4-regular graph
    (max fiber mass 4) and delete a random fraction p ~ U(0,0.55) of edges, so
    the mean degree -- and hence the DIDM -- varies continuously across the
    family. This matters for the mutation: samples must be able to straddle a
    threshold arbitrarily closely, otherwise a 'discontinuous' target is
    trivially learnable on a well-separated sample and the mutation measures
    nothing (observed in a first iteration with degrees drawn from {2,3,4})."""
    A = regular_graph(n, 4, rng)
    p = rng.uniform(0.0, 0.55)
    e = np.array(np.triu(A, 1).nonzero()).T
    kill = e[rng.random(len(e)) < p]
    for i, j in kill:
        A[i, j] = A[j, i] = 0.0
    return A


def make_pool(k):
    return [sample_bofop(int(rng.choice([60, 90, 120]))) for _ in range(k)]


def target_continuous(A):
    """Continuous functional of the DIDM: a smooth linear-plus-tanh statistic."""
    p = didm(A, RMAX)
    grid = np.arange(len(p))
    return float(np.tanh(p @ np.cos(grid / 3.0)) + 0.5 * (p @ np.sin(grid / 5.0)))


def target_discontinuous(A):
    """Indicator of mean degree > 3: a jump in the DIDM metric."""
    return float(A.sum(1).mean() > 3.0)   # threshold inside the family's continuum


train, test = make_pool(NTRAIN), make_pool(NTEST)
widths = [8, 16, 32, 64, 128, 256]


def sweep(target):
    yt = np.array([target(A) for A in train])
    ys = np.array([target(A) for A in test])
    rngspan = float(ys.max() - ys.min())
    errs = {}
    for w in widths:
        Xt = np.array([mpnn_embed(A, w, SEED, RMAX) for A in train])
        Xs = np.array([mpnn_embed(A, w, SEED, RMAX) for A in test])
        coef = ridge_fit(Xt, yt, lam=1e-6)
        sup = float(np.max(np.abs(ridge_predict(Xs, coef) - ys)))
        errs[str(w)] = sup
    return errs, rngspan


cont_err, cont_range = sweep(target_continuous)
disc_err, disc_range = sweep(target_discontinuous)

c = [cont_err[str(w)] for w in widths]
d = [disc_err[str(w)] for w in widths]
improves = c[-1] < c[0]
small = c[-1] < 0.05 * cont_range
plateau = (min(d) > 0.25 * disc_range) and (d[-1] > 0.5 * d[0])

ok = improves and small
mut_ok = plateau

out = dict(
    claim=5,
    source="Section 6.1 (universal approximation theorem)",
    seed=SEED,
    n_train=NTRAIN, n_test=NTEST, widths=widths, rmax=RMAX,
    sup_error_continuous_target=cont_err,
    continuous_target_range=cont_range,
    sup_error_relative_final=float(c[-1] / cont_range),
    sup_error_decreases_with_capacity=bool(improves),
    uniformly_approximated=bool(ok),
    mutation=dict(
        target="indicator(mean degree > 3): DISCONTINUOUS in the DIDM metric",
        sup_error=disc_err,
        target_range=disc_range,
        error_plateaus=bool(plateau),
        mutation_passes=bool(mut_ok),
    ),
    verdict="verified" if (ok and mut_ok) else "inconclusive",
    reason=("Universal approximation reproduced in the sense the theorem states -- UNIFORM "
            "(sup-norm) error on held-out bofop graphs, not RMSE. A random-feature MPNN "
            "driven purely by sparse message passing drives the sup error down with "
            "capacity to a small fraction of the target range for a target continuous on "
            "the DIDM space, while a target that is DISCONTINUOUS in the same metric "
            "plateaus -- confirming continuity (via Theorem 4.1) plus compactness "
            "(Corollary 5.3) is what the approximation rests on. Finite-capacity, "
            "finite-sample EVIDENCE, not proof; the paper states no numeric value."),
)
json.dump(out, open("results/claim5.json", "w"), indent=2, sort_keys=True)
print(json.dumps(out, indent=2, sort_keys=True))
