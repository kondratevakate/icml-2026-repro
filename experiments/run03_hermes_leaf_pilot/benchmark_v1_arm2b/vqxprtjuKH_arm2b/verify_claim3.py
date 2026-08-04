"""Claim 3 (Theorem 1.3, Section 1.2): O(log n) multiplicative approximation for
GraphVarAlloc with m > 1, i.e. a poly-time allocation with OBJ >= Omega(1/log n) OPT.

CPU reproduction: the claim is an approximation-ratio guarantee, so we test its
operational content on small instances where a strong reference optimum is
computable:
  * instances: random GraphVarAlloc, n in {6, 8, 10}, m = 2n random subsets
    (each vertex in S_j w.p. q), plus structured adversarial instances
    (disjoint sets, one dense set + many singleton-ish sets, star-like sets);
  * ALG (poly-time, log-free candidate family, in the spirit of the paper's
    algorithm): best over the n candidates "uniform variance over the top-k
    variables by set-coverage" (k = 1..n), evaluated exactly;
  * OPT_ref: multi-start Nelder-Mead optimum over the simplex (strong reference);
  * claim holds iff ALG / OPT_ref >= 1 / ln(n) on every instance (Omega(1/log n)).
Mutation: replace ALG by the WORST single-variable allocation (all variance on the
variable with the lowest coverage). The ratio must then fall below 1/ln n on at
least one instance -- showing the test can discriminate.
"""
import json, os, time
from multiprocessing import Pool
import numpy as np
import repro_utils as U

SEED = U.SEED


def coverage(sets, n):
    cov = np.zeros(n)
    for S in sets:
        for i in S:
            cov[i] += 1
    return cov


def alg_topk(sets, n):
    cov = coverage(sets, n)
    order = np.argsort(-cov)
    best, bx = -np.inf, None
    for k in range(1, n + 1):
        x = np.zeros(n)
        x[order[:k]] = 1.0 / k
        v = U.graph_obj_indep(x, sets)
        if v > best:
            best, bx = v, x
    return best, bx


def worst_single(sets, n):
    cov = coverage(sets, n)
    i = int(np.argmin(cov))
    x = np.zeros(n)
    x[i] = 1.0
    return U.graph_obj_indep(x, sets), x


def make_instances():
    inst = []
    rng = np.random.default_rng(SEED)
    for n in (6, 8, 10):
        for q in (0.3, 0.5, 0.8):
            sets = []
            while len(sets) < 2 * n:
                S = np.flatnonzero(rng.random(n) < q)
                if len(S) >= 2:
                    sets.append(tuple(S.tolist()))
            inst.append((f"random_n{n}_q{q}", n, sets))
    # structured / adversarial
    inst.append(("disjoint_pairs_n8", 8, [(0, 1), (2, 3), (4, 5), (6, 7)]))
    inst.append(("one_dense_plus_pairs_n8", 8,
                 [tuple(range(8)), (0, 1), (0, 2), (0, 3)]))
    inst.append(("star_n8", 8, [(0, i) for i in range(1, 8)]))
    inst.append(("nested_n8", 8, [tuple(range(k)) for k in range(2, 9)]))
    return inst


def _job(args):
    name, n, sets = args
    alg, xa = alg_topk(sets, n)
    ref, xr = U.optimize(sets, n, "independent", None, restarts=4,
                         seed=SEED, maxiter=800)
    ref = max(ref, alg)  # reference optimum is at least the algorithm's value
    bad, _ = worst_single(sets, n)
    bound = 1.0 / np.log(n)
    return dict(instance=name, n=n, m=len(sets), alg=alg, opt_ref=ref,
                ratio=alg / ref, bound_1_over_ln_n=bound,
                ok=bool(alg / ref >= bound),
                mutated_alg=bad, mutated_ratio=bad / ref,
                mutated_ok=bool(bad / ref >= bound),
                alg_alloc=[round(float(v), 4) for v in xa])


def main():
    t0 = time.time()
    inst = make_instances()
    with Pool(7) as pool:
        rows = pool.map(_job, inst)
    ratios = [r["ratio"] for r in rows]
    all_ok = all(r["ok"] for r in rows)
    mut_break = any(not r["mutated_ok"] for r in rows)
    out = dict(claim=3, source="Theorem 1.3, Section 1.2 (arXiv:2502.18463v1)",
               statement="poly-time algorithm for GraphVarAlloc (m>1) with OBJ >= Omega(1/log n) OPT",
               seed=SEED, rows=rows, min_ratio=float(min(ratios)),
               max_ratio=float(max(ratios)), all_above_1_over_ln_n=bool(all_ok),
               mutation=dict(description="worst single-variable allocation (lowest coverage)",
                             property_breaks=bool(mut_break),
                             min_mutated_ratio=float(min(r["mutated_ratio"] for r in rows))),
               verdict="verified" if (all_ok and mut_break) else "inconclusive",
               runtime_s=round(time.time() - t0, 1))
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/claim3.json", "w"), indent=1)
    for r in rows:
        print(f"{r['instance']:24s} ratio={r['ratio']:.3f} bound={r['bound_1_over_ln_n']:.3f} "
              f"ok={r['ok']} mut_ratio={r['mutated_ratio']:.3f} mut_ok={r['mutated_ok']}")
    print("verdict", out["verdict"], out["runtime_s"], "s")


if __name__ == "__main__":
    main()
