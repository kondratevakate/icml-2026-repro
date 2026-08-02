"""verify_claim3.py -- Theorem 3 (Section 3.2), asymptotic optimality of max-p aggregation
with learned conformity scores.

Claim under test (paper, Thm 3):
  With s_k(x,y) := -hhat(x,y), hhat = sum_k lambdahat_k(x) fhat_k(y|x), randomized
  p-values p^{(k)} and Chat^{(n)}(x) = {y : max_k p^{(k)}(x,y) >= alpha},
      limsup_n | |Chat^{(n)}| - |C*| |  <=  rho(T),   T = {(x,y) : h*(x,y) = 1},
  and if rho(T)=0 then rho(Chat^{(n)} triangle C*) -> 0 in probability.

Setup (fully discrete so that C* is computable exactly):
  X uniform on {0..nx-1} (nu = uniform), Y on {0..m-1} (mu = counting).
  True f_k(y|x) drawn once per world; the oracle lambda*(x) and C*(x) are obtained
  by solving the LP (4) exactly per x (same solver as verify_claim2.py).
  Estimation: fhat_k from multinomial training counts, lambdahat(x) by re-solving the
  LP on fhat -- both consistent, as Thm 3 assumes. rho(T) = 0 in these worlds
  (checked and reported).

Growing sample sizes n in {100, 400, 1600, 6400, 25600} per source (train = calib = n),
8 independent worlds (seeds), R randomized test evaluations per world.

Mutation test:
  MUT-lambda: freeze lambdahat at a uniform vector (deliberately inconsistent, violating
  sup_x||lambdahat - lambda*||->0). Prediction: the size gap / symmetric difference must
  NOT vanish with n, while finite-sample coverage (Thm 1) still holds.

Writes results/claim3.json.
"""
import json, os, time
import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.1
K, NX, M = 3, 5, 12
NS = [100, 400, 1600, 6400, 25600]
R = 120                       # randomized evaluations (U_k draws) per (world, n)
WORLDS = 8


def lp_lambda(F, alpha):
    K_, m = F.shape
    res = linprog(np.ones(m), A_ub=-F, b_ub=-np.ones(K_) * (1 - alpha),
                  bounds=[(0.0, 1.0)] * m, method="highs")
    assert res.status == 0
    return -np.asarray(res.ineqlin.marginals), float(res.fun)


def make_world(rng):
    f = rng.gamma(0.9, 1.0, size=(K, NX, M)) + 1e-3
    f /= f.sum(axis=2, keepdims=True)
    return f


def oracle(f):
    """Oracle lambda*, h*, the core set {h*>1}, the tie/boundary set T and rho(T).

    Sizes follow the paper's convention |C| = int mu(C(x)) dnu(x), i.e. here the
    average number of included labels per x (nu = uniform on the X grid).
    """
    lam = np.zeros((NX, K)); core = np.zeros((NX, M)); hstar = np.zeros((NX, M))
    ties = np.zeros((NX, M)); lp_val = 0.0
    for x in range(NX):
        l, val = lp_lambda(f[:, x, :], ALPHA)
        lam[x] = l
        h = l @ f[:, x, :]
        hstar[x] = h
        core[x] = (h > 1.0 + 1e-9).astype(float)
        ties[x] = (np.abs(h - 1.0) <= 1e-9).astype(float)
        lp_val += val
    rho_T = ties.sum() / NX
    return lam, hstar, core, ties, core.sum() / NX, rho_T, lp_val / NX


def sample(rng, f, k, n):
    x = rng.integers(0, NX, size=n)
    cdf = np.cumsum(f[k], axis=1)
    y = np.minimum((rng.random(n)[:, None] > cdf[x]).sum(axis=1), M - 1)
    return x, y


def run_world(seed, mutate="none"):
    rng = np.random.default_rng(seed)
    f = make_world(rng)
    lam_star, h_star, core, ties, size_core, rho_T, lp_size = oracle(f)
    out = []
    for n in NS:
        # ---- training fold: estimate fhat and lambdahat --------------------
        counts = np.ones((K, NX, M)) * 1e-6
        for k in range(K):
            tx, ty = sample(rng, f, k, n)
            np.add.at(counts[k], (tx, ty), 1.0)
        fhat = counts / counts.sum(axis=2, keepdims=True)
        hhat = np.zeros((NX, M))
        for x in range(NX):
            if mutate == "uniform_lambda":
                lh = np.full(K, float(lam_star[x].sum()) / K)   # inconsistent lambda
                hhat[x] = lh @ fhat[:, x, :]
            elif mutate == "single_density":
                # score built from ONE source density only: violates the
                # "consistent for -h*" premise of Thm 3
                hhat[x] = fhat[0, x, :]
            else:
                lh, _ = lp_lambda(fhat[:, x, :], ALPHA)
                hhat[x] = lh @ fhat[:, x, :]
        # ---- calibration fold ----------------------------------------------
        cal = []
        for k in range(K):
            cx, cy = sample(rng, f, k, n)
            cal.append(-hhat[cx, cy])
        # ---- prediction sets over the whole X grid, R randomizations --------
        size_acc = 0.0; symdiff_acc = 0.0; symdiff_nontie_acc = 0.0
        cov_hit = np.zeros(K); cov_n = np.zeros(K)
        for r in range(R):
            C = np.zeros((NX, M))
            P = np.zeros((K, NX, M))
            for k in range(K):
                S = np.sort(cal[k]); nk = S.size
                s = -hhat                                  # (NX,M) candidate scores
                gt = nk - np.searchsorted(S, s, side="right")
                eq = np.searchsorted(S, s, side="right") - np.searchsorted(S, s, side="left")
                U = rng.random()
                P[k] = (gt + (1 + eq) * U) / (nk + 1.0)
            pagg = P.max(axis=0)
            C = (pagg >= ALPHA).astype(float)
            size_acc += C.sum() / NX
            symdiff_acc += np.abs(C - core).sum() / NX
            # Thm 3 only controls the set OUTSIDE the boundary T = {h*=1}
            symdiff_nontie_acc += (np.abs(C - core) * (1.0 - ties)).sum() / NX
            # coverage check (Thm 1 must survive the mutation)
            for k in range(K):
                nt = 50
                x0 = rng.integers(0, NX, size=nt)
                cdf = np.cumsum(f[k], axis=1)
                y0 = np.minimum((rng.random(nt)[:, None] > cdf[x0]).sum(axis=1), M - 1)
                cov_hit[k] += C[x0, y0].sum(); cov_n[k] += nt
        mean_size = size_acc / R
        out.append(dict(n=n, mean_size=mean_size,
                        oracle_core_size=size_core, lp_optimal_size=lp_size, rho_T=rho_T,
                        abs_size_gap_vs_core=abs(mean_size - size_core),
                        thm3_bound_satisfied=bool(abs(mean_size - lp_size) <= rho_T + 1e-9),
                        slack_vs_thm3_bound=float(rho_T - abs(mean_size - lp_size)),
                        mean_symdiff=symdiff_acc / R,
                        mean_symdiff_outside_T=symdiff_nontie_acc / R,
                        worst_source_coverage=float((cov_hit / cov_n).min())))
    return dict(seed=seed, rho_T=rho_T, oracle_core_size=size_core,
                lp_optimal_size=lp_size, per_n=out)


def summarize(worlds):
    res = {}
    for i, n in enumerate(NS):
        g = np.array([w["per_n"][i]["abs_size_gap_vs_core"] for w in worlds])
        s = np.array([w["per_n"][i]["mean_symdiff_outside_T"] for w in worlds])
        c = np.array([w["per_n"][i]["worst_source_coverage"] for w in worlds])
        b = np.array([w["per_n"][i]["thm3_bound_satisfied"] for w in worlds])
        res[str(n)] = dict(mean_abs_size_gap_vs_core=float(g.mean()), max_abs_size_gap_vs_core=float(g.max()),
                           mean_symdiff_outside_T=float(s.mean()), max_symdiff_outside_T=float(s.max()),
                           n_worlds_satisfying_thm3_bound=int(b.sum()), n_worlds=len(worlds),
                           min_worst_source_coverage=float(c.min()))
    return res


def main():
    t0 = time.time()
    seeds = [7000 + i for i in range(WORLDS)]
    main_w = [run_world(s, mutate="none") for s in seeds]
    mut_w = [run_world(s, mutate="uniform_lambda") for s in seeds]
    mut2_w = [run_world(s, mutate="single_density") for s in seeds]
    out = dict(
        claim="Theorem 3 (Section 3.2): asymptotic optimality of max-p aggregation with learned scores",
        command="python verify_claim3.py",
        alpha=ALPHA, K=K, n_x=NX, m=M, sample_sizes=NS, worlds=WORLDS, randomizations=R,
        rho_T_mean=float(np.mean([w["rho_T"] for w in main_w])),
        rho_T_per_world=[w["rho_T"] for w in main_w],
        main=summarize(main_w), mutation_frozen_uniform_lambda=summarize(mut_w),
        mutation_single_source_density_score=summarize(mut2_w),
        main_worlds=main_w, mutation_worlds=mut_w, mutation2_worlds=mut2_w,
        runtime_sec=round(time.time() - t0, 1))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "claim3.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items() if "worlds" not in k or k == "worlds"}, indent=2))


if __name__ == "__main__":
    main()
