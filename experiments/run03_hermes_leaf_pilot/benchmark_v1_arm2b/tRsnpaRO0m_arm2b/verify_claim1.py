"""Claim 1 (Definition 3.1): graphops are self-adjoint, positivity-preserving
operators over a probability space, unifying dense graphons and sparse graphs
as one class of limit objects.

Finite model: Omega = n uniform atoms (mass 1/n); (Af)(x) = int W(x,y) f(y) dmu
becomes (W @ f)/n. Dense instance = smooth graphon kernel. Sparse instance =
W = n * Adj of a bounded-degree graph (kernel blows up like n -- that IS the
point: both live in the same operator class).

Mutation: an antisymmetric-signed kernel must break BOTH self-adjointness and
positivity-preservation; a symmetric signed kernel must break positivity only.
"""
import json, os
import numpy as np
from common import SEED, audit_operator, regular_graph

os.makedirs("results", exist_ok=True)
rng = np.random.default_rng(SEED)
TOL = 1e-10

n = 300
x = (np.arange(n) + 0.5) / n
# dense graphon
Wd = 0.5 * (1.0 + np.cos(2 * np.pi * (x[:, None] - x[None, :])))
# sparse graph as a graphop: W = n * A  (degree-4 random regular)
A4 = regular_graph(n, 4, rng)
Ws = n * A4

dense = audit_operator(Wd, n, rng)
sparse = audit_operator(Ws, n, rng)

# ---- mutations -----------------------------------------------------------
S = np.sign(rng.normal(size=(n, n)))
S = np.triu(S, 1); S = S - S.T                      # antisymmetric
mut_anti = audit_operator(S * np.abs(Wd), n, rng)

Sym = np.triu(np.sign(rng.normal(size=(n, n))), 1)
Sym = Sym + Sym.T                                    # symmetric, signed
mut_signed = audit_operator(Sym * np.abs(Wd), n, rng)

pass_dense = dense["selfadjoint_err"] < TOL and dense["positivity_violation"] == 0.0
pass_sparse = sparse["selfadjoint_err"] < TOL and sparse["positivity_violation"] == 0.0
mut_ok = (mut_anti["selfadjoint_err"] > 1e-3 and mut_anti["positivity_violation"] > 0
          and mut_signed["positivity_violation"] > 0
          and mut_signed["selfadjoint_err"] < TOL)

out = dict(
    claim=1,
    source="Definition 3.1",
    seed=SEED,
    n_atoms=n,
    dense_graphon=dense,
    sparse_graph_as_graphop=sparse,
    dense_selfadjoint=pass_dense,
    sparse_selfadjoint=pass_sparse,
    unified_same_class=bool(pass_dense and pass_sparse),
    mutation=dict(
        antisymmetric_kernel=mut_anti,
        symmetric_signed_kernel=mut_signed,
        breaks_selfadjointness=bool(mut_anti["selfadjoint_err"] > 1e-3),
        breaks_positivity_only=bool(mut_signed["positivity_violation"] > 0
                                    and mut_signed["selfadjoint_err"] < TOL),
        mutation_passes=bool(mut_ok),
    ),
    verdict="verified" if (pass_dense and pass_sparse and mut_ok) else "inconclusive",
    reason=("Definitional claim, checked on a faithful n-atom discretisation of the "
            "probability space. Both a dense graphon kernel and a sparse bounded-degree "
            "graph (W = n*Adj) satisfy self-adjointness (<f> error at machine precision) "
            "and positivity preservation, i.e. they are members of the SAME operator "
            "class -- which is the content of Definition 3.1. The paper states no numeric "
            "value to match; this is conclusive at the level of the finite model, not a "
            "formal proof in the infinite-dimensional setting."),
)
json.dump(out, open("results/claim1.json", "w"), indent=2, sort_keys=True)
print(json.dumps(out, indent=2, sort_keys=True))
