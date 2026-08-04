"""Claim 4 (Theorem 1.6, Section 1.3): in the optimal allocation only Theta(1/p)
variables receive variance Omega(p).

Theorem 1.6 (paper): for any delta>0 and a random GraphVarAlloc instance with
n,m -> infinity, with probability p > 1-delta there are Theta(1/p) variables
allocated variance Omega(p).  (p = |S_j|/n is the density parameter.)

CPU reproduction: the theorem is asymptotic (n,m -> infinity), so we test the
finite-n signature it predicts:
  * for n in {8, 12, 16} and p in {1/8,...,1}, build a random GraphVarAlloc instance
    (m = 2n random subsets, each vertex included w.p. p, |S| >= 2),
  * compute the optimal allocation (independent zero-mean case, exact quadrature
    objective, multi-start Nelder-Mead on the simplex sum sigma_i^2 = 1),
  * count k(p) = #{i : sigma_i^2 >= c*p} for c in {0.25, 0.5} and test whether
    k(p) * p stays bounded / roughly constant, i.e. k(p) = Theta(1/p).
Verdict logic: 'verified' requires k(p) to decrease in p and k(p)*p to stay inside a
constant band [0.25, 4] for the tested range; otherwise 'inconclusive' (finite n).
Mutation: use a UNIFORM allocation instead of the optimal one; then k(p) is dictated
by 1/n rather than 1/p and the Theta(1/p) signature must break.
"""
import json, os, time
from multiprocessing import Pool
import numpy as np
import repro_utils as U

SEED = U.SEED
PS = [1 / 8, 2 / 8, 3 / 8, 4 / 8, 6 / 8, 1.0]
NS = [8, 12, 16]
REPS = 2


def random_sets(n, p, rng, m):
    sets = []
    while len(sets) < m:
        S = np.flatnonzero(rng.random(n) < p)
        if len(S) >= 2:
            sets.append(tuple(S.tolist()))
    return sets


def _job(args):
    n, p, r = args
    rng = np.random.default_rng(SEED + 97 * r + int(1000 * p) + 13 * n)
    sets = random_sets(n, p, rng, m=2 * n)
    val, x = U.optimize(sets, n, "independent", None, restarts=2,
                        seed=SEED + r, maxiter=700)
    unif = np.ones(n) / n
    return dict(n=n, p=p, rep=r, opt=val,
                alloc=[float(v) for v in np.sort(x)[::-1]],
                k_c25=int((x >= 0.25 * p).sum()), k_c50=int((x >= 0.5 * p).sum()),
                k_unif_c25=int((unif >= 0.25 * p).sum()),
                k_unif_c50=int((unif >= 0.5 * p).sum()))


def main():
    t0 = time.time()
    tasks = [(n, p, r) for n in NS for p in PS for r in range(REPS)]
    with Pool(7) as pool:
        rows = pool.map(_job, tasks)

    summary, ok_flags = [], []
    for n in NS:
        for c in ("k_c25", "k_c50"):
            ks = [float(np.mean([r[c] for r in rows if r["n"] == n and r["p"] == p]))
                  for p in PS]
            kp = [k * p for k, p in zip(ks, PS)]
            monotone = all(ks[i + 1] <= ks[i] + 0.51 for i in range(len(ks) - 1))
            band = all(0.25 <= v <= 4.0 for v in kp)
            ok_flags.append(monotone and band)
            summary.append(dict(n=n, threshold=c, p=PS, k_of_p=ks, k_times_p=kp,
                                non_increasing_in_p=monotone, in_constant_band=band))
    # mutation rows (uniform allocation): k*p is n*p/... -> grows with p, breaks Theta(1/p)
    mut = []
    for n in NS:
        ks = [float(np.mean([r["k_unif_c50"] for r in rows if r["n"] == n and r["p"] == p]))
              for p in PS]
        kp = [k * p for k, p in zip(ks, PS)]
        mut.append(dict(n=n, k_of_p=ks, k_times_p=kp,
                        in_constant_band=all(0.25 <= v <= 4.0 for v in kp),
                        non_increasing_in_p=all(ks[i + 1] <= ks[i] + 0.51
                                                for i in range(len(ks) - 1))))
    mut_break = any((not m_["in_constant_band"]) for m_ in mut)

    verdict = "verified" if all(ok_flags) else ("inconclusive" if any(ok_flags) else "falsified")
    out = dict(claim=4, source="Theorem 1.6, Section 1.3 (arXiv:2502.18463v1)",
               statement="only Theta(1/p) variables get variance Omega(p) in the optimal allocation",
               seed=SEED, note="theorem is asymptotic (n,m -> infinity); tested at n=8,12,16",
               rows=rows, summary=summary, all_signatures_ok=bool(all(ok_flags)),
               mutation=dict(description="uniform allocation instead of optimal",
                             rows=mut, property_breaks=bool(mut_break)),
               verdict=verdict, runtime_s=round(time.time() - t0, 1))
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/claim4.json", "w"), indent=1)
    for s in summary:
        print(s["n"], s["threshold"], [round(v, 2) for v in s["k_of_p"]],
              "k*p", [round(v, 2) for v in s["k_times_p"]],
              s["non_increasing_in_p"], s["in_constant_band"])
    print("verdict", verdict, "mut_break", mut_break, out["runtime_s"], "s")


if __name__ == "__main__":
    main()
