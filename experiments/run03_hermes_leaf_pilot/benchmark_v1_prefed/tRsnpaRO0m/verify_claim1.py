"""Claim 1 (Definition 3.1): graphops are self-adjoint, positivity-preserving
operators on a probability space, and BOTH dense graphons and sparse graphs
are instances of the same class.

Finite-dimensional faithful discretisation: Omega = [0,1] with Lebesgue measure,
partitioned into n intervals of mass 1/n. An operator A given by a symmetric
kernel W >= 0 acts as (Af)(x) = int W(x,y) f(y) dmu(y)  ->  (1/n) W f.

Checks:
  P1 self-adjointness:      <Af,g>_mu == <f,Ag>_mu
  P2 positivity-preserving: f >= 0  =>  Af >= 0
  P3 boundedness L^inf -> L^1
  P4 both a dense graphon sample and a sparse (bounded-degree) graph
     produce operators satisfying P1-P3, i.e. one common class.
Mutation: antisymmetric kernel breaks P1; signed kernel breaks P2.
"""
import json, os
import numpy as np

SEED = 20260802
rng = np.random.default_rng(SEED)
os.makedirs("results", exist_ok=True)


def op(W, n):
    """graphop action on the uniform probability space with n atoms"""
    return lambda f: (W @ f) / n


def inner(f, g, n):
    return float(np.dot(f, g) / n)


def audit(W, n, trials=200, rng=rng):
    A = op(W, n)
    sa, pos = 0.0, 0.0
    for _ in range(trials):
        f = rng.normal(size=n)
        g = rng.normal(size=n)
        sa = max(sa, abs(inner(A(f), g, n) - inner(f, A(g), n)))
        h = rng.random(n)                      # h >= 0
        pos = max(pos, float(max(0.0, -np.min(A(h)))))
    # ||A f||_1 for ||f||_inf <= 1  (worst case f = sign row-wise -> all ones)
    f1 = np.ones(n)
    norm_inf_to_1 = float(np.sum(np.abs(A(f1))) / n)
    return dict(selfadjoint_err=sa, positivity_violation=pos,
                op_norm_inf_to_1=norm_inf_to_1)


n = 400

# --- dense graphon instance: W(x,y) = 0.5*(1 + cos(2*pi*(x-y))) sampled -----
x = (np.arange(n) + 0.5) / n
Wd = 0.5 * (1 + np.cos(2 * np.pi * (x[:, None] - x[None, :])))
dense = audit(Wd, n)

# --- sparse instance: 3-regular-ish random graph, kernel scaled by n --------
deg = 3
Asp = np.zeros((n, n))
for i in range(n):
    for j in rng.choice(n, size=deg, replace=False):
        if i != j:
            Asp[i, j] = Asp[j, i] = 1.0
Ws = n * Asp        # sparse graph as a graphop: kernel blows up like n
sparse = audit(Ws, n)

# --- mutations --------------------------------------------------------------
Wanti = Wd - Wd.T + rng.normal(size=(n, n)) * 0.0
Wanti = np.triu(Wd, 1); Wanti = Wanti - Wanti.T          # antisymmetric
mut_selfadj = audit(Wanti, n)

Wsigned = Wd - 1.0                                        # takes negative values
mut_pos = audit(Wsigned, n)

TOL = 1e-9
verified = (dense["selfadjoint_err"] < TOL and dense["positivity_violation"] < TOL
            and sparse["selfadjoint_err"] < TOL and sparse["positivity_violation"] < TOL
            and np.isfinite(dense["op_norm_inf_to_1"]) and np.isfinite(sparse["op_norm_inf_to_1"]))
mutation_breaks = (mut_selfadj["selfadjoint_err"] > 1e-3
                   and mut_pos["positivity_violation"] > 1e-3)

out = dict(
    claim=1, source="Definition 3.1", seed=SEED, n=n,
    dense_graphon=dense, sparse_graph=sparse,
    mutation_antisymmetric=mut_selfadj, mutation_signed_kernel=mut_pos,
    tolerance=TOL,
    verdict="verified" if (verified and mutation_breaks) else "inconclusive",
    note=("Both a dense graphon kernel and a sparse bounded-degree graph, mapped to "
          "operators on the same probability space, are self-adjoint and positivity-"
          "preserving with finite L^inf->L^1 norm; mutations break exactly the "
          "property they target. This is a definitional/structural check, not an "
          "empirical result of the paper."))
print(json.dumps(out, indent=2))
json.dump(out, open("results/claim1.json", "w"), indent=2)
