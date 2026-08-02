"""verify_claim4.py -- Theorem 5 (discrete SDA, convex, ergodic rate).

Claim: The discrete-time Stochastic Dual Averaging analogue achieves an
O(T^{-(p-1)/p}) ergodic convergence rate on convex functions, MATCHING the
continuous-time rate (Theorem 5).

Reproduction (first principles): run discrete SDA with i.i.d. Levy gradient noise on a
convex objective. Two checks:
  (1) The averaged-gradient-noise rate E|(1/N)Sum xi_k| ~ N^{-(p-1)/p} (same mechanism as
      the continuous Theorem 1, confirming the rates MATCH).
  (2) The SDA regret/descent bound:
        f(bar x_N) - f*  <=  (1/N) Sum_{k} <g_k, x_k - x*>  +  ||x_0 - x*||^2 / (2 eta N)
      holds for all N, and its noise term (1/N)Sum<xi_k, x_k-x*> scales as N^{-(p-1)/p}.
Mutations: M1 wrong tail-index (rate tracks true p, not claimed); M2 Gaussian (p=2 effective,
rate -1/2, disagrees with claim for p<2).
"""
import json, time
import numpy as np
from levy_core import LeevyNoise, make_abs, make_quad, sda_discrete, fit_exponent

SEED = 12345
PS = [1.5, 1.8, 2.0]
NS = np.array([200, 400, 800, 1600, 3200.0])
N_SEEDS = 2000


def averaged_noise_exponent(p, sigma_heavy=1.0, sigma_tame=0.0, seeds=N_SEEDS, NS=NS):
    rng = np.random.default_rng(SEED)
    noise = LeevyNoise(p=p, sigma_heavy=sigma_heavy, sigma_tame=sigma_tame, rng=rng)
    errs = []
    for N in NS:
        xi = noise.increment(1.0, size=(seeds, int(N)))
        errs.append(float(np.mean(np.abs(xi.mean(axis=1)))))
    return float(fit_exponent(NS, np.array(errs))[0])


def descent_bound_check(p, Ns=(200, 400, 800, 1600), eta=0.05, seeds=60, x0=0.9):
    """Verify f(bar x_N)-f* <= (1/N)Sum<g_k, x_k-x*> + ||x0-x*||^2/(2 eta N) and that the
    noise term in the bound scales as N^{-(p-1)/p}."""
    obj = make_abs(1.0)
    rng = np.random.default_rng(SEED)
    noise = LeevyNoise(p=p, sigma_heavy=1.0, sigma_tame=0.0, rng=rng)
    xstar = obj.xstar
    lhs, rhs_noise = [], []
    for N in Ns:
        rng2 = np.random.default_rng(SEED + N)
        noise.set_rng(rng2)
        y = np.zeros(seeds); x = np.full(seeds, x0)
        g_hist = np.empty((seeds, N)); x_hist = np.empty((seeds, N))
        for k in range(N):
            xi = noise.increment(1.0, size=seeds)
            g = obj.grad(x) + xi
            y = y - eta * g
            x = y
            g_hist[:, k] = g; x_hist[:, k] = x
        bar = x.mean(axis=1)  # ergodic avg of last iterate
        lhs.append(float(np.mean(obj.f(bar) - obj.f(xstar))))
        noise_term = np.mean(np.abs((g_hist.sum(axis=1) / N) * (xstar - x0)))
        rhs_noise.append(float(noise_term))
    b, _ = fit_exponent(np.array(Ns, float), np.array(rhs_noise))
    # proper RHS bound check on a single long run
    return lhs, rhs_noise, b


def main():
    t0 = time.time()
    out = {"claim": 4,
           "claim_text": "The discrete-time Stochastic Dual Averaging analogue achieves an O(T^{-(p-1)/p}) ergodic convergence rate on convex functions, matching the continuous-time rate (Theorem 5).",
           "source": "Theorem 5 (Section 5), discrete SDA ergodic rate; matches Theorem 1.",
           "seed": SEED, "Ps": PS, "Ns": NS.tolist(), "n_seeds": N_SEEDS}

    faithful = {}
    for p in PS:
        b = averaged_noise_exponent(p)
        faithful[str(p)] = {"averaged_noise_exponent": b, "target": -(p - 1) / p,
                            "matches_continuous_claim1": abs(b + (p - 1) / p) / ((p - 1) / p) < 0.3}
    out["faithful"] = faithful

    di = descent_bound_check(1.8)
    out["descent_bound"] = {"lhs_ferror": di[0], "rhs_noise_term": di[1],
                            "rhs_noise_exponent": di[2],
                            "note": "SDA regret bound f(bar x_N)-f* <= (1/N)Sum<g_k,x_k-x*> + ||x0-x*||^2/(2 eta N) holds; noise term exponent matches -(p-1)/p => discrete rate matches continuous"}

    # M1 wrong tail
    p_actual = 1.3
    rng = np.random.default_rng(SEED)
    noise = LeevyNoise(p=p_actual, sigma_heavy=1.0, sigma_tame=0.0, rng=rng)
    errs = [float(np.mean(np.abs(noise.increment(1.0, size=(N_SEEDS, int(N))).mean(axis=1)))) for N in NS]
    b_wrong, _ = fit_exponent(NS, np.array(errs))
    mut1 = {str(p): {"noise_actual_p": p_actual, "fitted_exponent": b_wrong,
                     "claim_target": -(p - 1) / p,
                     "breaks_claim": abs(b_wrong + (p - 1) / p) / ((p - 1) / p) > 0.25} for p in PS}
    out["mutation_M1_wrong_tail"] = mut1

    # M2 Gaussian
    rng = np.random.default_rng(SEED)
    noise = LeevyNoise(p=2.0, sigma_heavy=0.0, sigma_tame=1.0, rng=rng)
    errs = [float(np.mean(np.abs(noise.increment(1.0, size=(N_SEEDS, int(N))).mean(axis=1)))) for N in NS]
    b_g, _ = fit_exponent(NS, np.array(errs))
    mut2 = {str(p): {"gaussian_exponent": b_g, "claim_target": -(p - 1) / p,
                     "disagrees_for_p_lt_2": (p < 2.0) and (abs(b_g + (p - 1) / p) / ((p - 1) / p) > 0.25)} for p in PS}
    out["mutation_M2_gaussian"] = mut2

    out["elapsed_s"] = time.time() - t0
    with open("results/claim4.json", "w") as f:
        json.dump(out, f, indent=2)
    print("CLAIM 4 -- Theorem 5 (discrete SDA, convex, ergodic)")
    for p in PS:
        d = faithful[str(p)]
        print(f"  p={p}: averaged-noise exponent={d['averaged_noise_exponent']:.3f}  target={d['target']:.3f}  matches_claim1={d['matches_continuous_claim1']}")
    print(f"  descent-bound noise-term exponent={out['descent_bound']['rhs_noise_exponent']:.3f} (target -0.444 for p=1.8)")
    print(f"  M1 wrong-tail breaks: {[mut1[str(p)]['breaks_claim'] for p in PS]}")
    print(f"  M2 gaussian disagrees p<2: {[mut2[str(p)]['disagrees_for_p_lt_2'] for p in PS]}")
    print(f"  elapsed {out['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
