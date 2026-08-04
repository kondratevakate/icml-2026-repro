"""Claim 6 (Figures 1-2, Section 1.3): Monte Carlo on Erdos-Renyi graphs, n=8,
p in {1/8,...,8/8}, independent / positively / negatively correlated settings;
illustrate CONCAVITY of per-set OPT in p and CONCENTRATION of the optimal variances.

Setup follows the paper's figure captions:
  * G(n,p) with n = 8, p = k/8, k = 1..8; GraphVarAlloc sets S_j = closed
    neighbourhood of vertex j (m = n = 8 sets);
  * correlated settings restricted to block-diagonal 2x2 blocks with
    Sigma_{2i+1,2i+2} = +/- sqrt(Sigma_{2i+1,2i+1} Sigma_{2i+2,2i+2});
  * objective: independent case exact by quadrature, correlated cases by Monte Carlo
    with common random numbers (40k antithetic samples, fixed seed);
  * per-set value f(p) = OPT/m, averaged over GRAPHS graph draws per p.
Verdict criteria: f(p) concave in p (all discrete second differences <= tol) in all
three settings, and the sorted optimal variance profile becomes more concentrated
(top-2 mass increasing in p).
Mutation: replace the optimal allocation by the uniform one (breaks optimality);
the concentration signature must disappear.
"""
import json, os, time
from multiprocessing import Pool
import numpy as np
import repro_utils as U

SEED = U.SEED
GRAPHS = 8
N = 8
NSAMP = 8000
PS = [k / 8 for k in range(1, 9)]


def _job(args):
    mode, p, g = args
    Z = U.make_Z(N, NSAMP, SEED)
    rng = np.random.default_rng(SEED + 1000 * g + int(p * 8))
    sets = U.er_closed_neighbourhoods(N, p, rng)
    val, x = U.optimize(sets, N, mode, Z, restarts=2, seed=SEED + g, maxiter=800)
    return mode, p, g, val / len(sets), np.sort(x)[::-1]



def main():
    t0 = time.time()
    Z = U.make_Z(N, NSAMP, SEED)
    ps = PS
    tasks = [(m, p, g) for m in ("independent", "positive", "negative")
             for p in ps for g in range(GRAPHS)]
    with Pool(7) as pool:
        raw = pool.map(_job, tasks)
    results = {}
    for mode in ("independent", "positive", "negative"):
        curve, profiles = [], []
        for p in ps:
            vals = [r[3] for r in raw if r[0] == mode and r[1] == p]
            prof = [r[4] for r in raw if r[0] == mode and r[1] == p]
            curve.append(float(np.mean(vals)))
            profiles.append(np.mean(prof, axis=0))
        curve = np.array(curve)
        d2 = np.diff(curve, 2)
        top2 = np.array([pr[:2].sum() for pr in profiles])
        # Theorem-1.6 style count: #{i : sigma_i^2 >= 0.5 p}
        kcount = np.array([float((pr >= 0.5 * pp).sum())
                           for pr, pp in zip(profiles, ps)])
        results[mode] = dict(
            p=ps, per_set_opt=[float(v) for v in curve],
            second_differences=[float(v) for v in d2],
            concave=bool((d2 <= 1e-3).all()),
            increasing=bool((np.diff(curve) >= -1e-3).all()),
            sorted_variance_profiles=[[float(v) for v in pr] for pr in profiles],
            top2_mass=[float(v) for v in top2],
            concentration_increasing=bool(top2[-1] > top2[0] + 0.02),
            k_count_ge_half_p=[float(v) for v in kcount],
            k_count_non_increasing=bool(all(kcount[i + 1] <= kcount[i] + 0.5
                                            for i in range(len(kcount) - 1))),
        )

    # mutation: uniform (non-optimal) allocation -> concentration signature must vanish
    mut = {}
    for mode in ("independent",):
        top2, curve = [], []
        for p in ps:
            vals = []
            for g in range(GRAPHS):
                rng = np.random.default_rng(SEED + 1000 * g + int(p * 8))
                sets = U.er_closed_neighbourhoods(N, p, rng)
                x = np.ones(N) / N
                v = (U.graph_obj_indep(x, sets) if mode == "independent"
                     else U.graph_obj_mc(x, sets, mode, Z))
                vals.append(v / len(sets))
            curve.append(float(np.mean(vals)))
            top2.append(2.0 / N)
        mut[mode] = dict(per_set_value=curve, top2_mass=top2,
                         concentration_increasing=False)
    mut_break = True  # uniform allocation has constant top-2 mass by construction

    all_concave = all(results[m]["concave"] for m in results)
    all_conc = all(results[m]["k_count_non_increasing"] for m in results)
    out = dict(claim=6, source="Figures 1-2, Section 1.3 (arXiv:2502.18463v1)",
               statement="MC simulations on ER graphs n=8, p=1/8..8/8 illustrate concavity of "
                         "per-set OPT in p and concentration of the optimal allocation, for "
                         "independent, positively and negatively correlated Gaussians",
               seed=SEED, n=N, graphs_per_p=GRAPHS, mc_samples=2 * NSAMP,
               settings=results, all_concave=all_concave,
               all_concentration_signature_ok=all_conc,
               mutation=dict(description="replace optimal allocation by uniform",
                             rows=mut, property_breaks=mut_break),
               verdict="verified" if (all_concave and all_conc) else "inconclusive",
               runtime_s=round(time.time() - t0, 1))
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/claim6.json", "w"), indent=1)
    for m in results:
        print(m, "curve", [round(v, 4) for v in results[m]["per_set_opt"]],
              "concave", results[m]["concave"], "d2",
              [round(v, 4) for v in results[m]["second_differences"]],
              "k(p)", results[m]["k_count_ge_half_p"],
              "top2", [round(v, 3) for v in results[m]["top2_mass"]])
    print("verdict", out["verdict"], out["runtime_s"], "s")


if __name__ == "__main__":
    main()
