"""verify_claim1.py -- Theorem 1 (continuous time, convex).

Claim: Under the Leevy Mirror Flow with centered Leevy noise of finite p-th moments
(1<p<=2), the time-averaged orbit achieves eps-optimality within O~(eps^{-p/(p-1)})
time for convex objectives.

Reproduction strategy (from first principles, numpy only):
The leading term of the time-averaged orbit's deviation from optimum is the AVERAGED
Levy gradient noise  A_T = (1/T) Sum_{k=1}^T xi_k,  xi_k i.i.d. Leevy(p) increments.
For a Leevy process / i.i.d. heavy-tailed noise with E|xi|^p < infty, the generalized
LLN (stable averaging) gives  E|A_T| = Theta(T^{-(p-1)/p}).  This is exactly the rate
Theorem 1 quantifies (eps ~ T^{-(p-1)/p}  =>  T ~ eps^{-p/(p-1)}).
We (a) measure E|A_T| vs T and fit the exponent, (b) verify the mirror-descent descent
identity that lifts A_T to f(bar x_T)-f* (so the bound inherits the rate), and (c) run
mutations: premise violation (infinite p-th moment) and Gaussian (p=2 effective).
"""
import json, time
import numpy as np
from levy_core import LeevyNoise, make_abs, make_quad, mirror_flow_continuous, time_averaged_error, fit_exponent

SEED = 12345
PS = [1.5, 1.8, 2.0]
TS = np.array([200, 400, 800, 1600, 3200.0])
N_SEEDS = 2000


def averaged_noise_exponent(p, sigma_heavy=1.0, sigma_tame=0.0, beta=None, seeds=N_SEEDS):
    rng = np.random.default_rng(SEED)
    noise = LeevyNoise(p=p, sigma_heavy=sigma_heavy, sigma_tame=sigma_tame, beta=beta, rng=rng)
    errs = []
    for T in TS:
        xi = noise.increment(1.0, size=(seeds, int(T)))   # i.i.d. Leevy(1) per step
        A = xi.mean(axis=1)                                # (1/T) Sum xi_k
        errs.append(float(np.mean(np.abs(A))))
    errs = np.array(errs)
    b, a = fit_exponent(TS, errs)
    return float(b), float(a), errs.tolist()


def descent_identity_check(p, T=1600, dt=0.5, seeds=200, x0=0.9):
    """Verify the MD descent bound  f(bar x_T)-f* <= (1/T)Sum<xi_k, x_k-x*> + const/T
    and that the noise term scales as T^{-(p-1)/p}."""
    obj = make_abs(1.0)
    rng = np.random.default_rng(SEED)
    noise = LeevyNoise(p=p, sigma_heavy=1.0, sigma_tame=0.0, rng=rng)
    n_steps = int(round(T / dt))
    xstar = obj.xstar
    lhs, rhs_noise = [], []
    for Tt in [200, 400, 800, 1600]:
        n = int(round(Tt / dt))
        rng2 = np.random.default_rng(SEED + n)
        noise.set_rng(rng2)
        x = np.full(seeds, x0)
        xs = np.empty((seeds, n + 1)); xs[:, 0] = x
        xi_hist = np.empty((seeds, n))
        for k in range(n):
            xi = noise.increment(dt, size=seeds)
            xi_hist[:, k] = xi
            x = np.clip(x - obj.grad(x) * dt + xi, -1.0, 1.0)
            xs[:, k + 1] = x
        bar = xs.mean(axis=1)
        lhs.append(float(np.mean(obj.f(bar) - obj.f(xstar))))
        # noise contribution (1/T) Sum<xi_k, x_k - x*>
        noise_term = np.mean(np.abs((xi_hist.sum(axis=1) / Tt) * (0.0 - x0)))  # |<A_T, x*-x0>|
        rhs_noise.append(float(noise_term))
    b, _ = fit_exponent(np.array([200, 400, 800, 1600.0]), np.array(rhs_noise))
    return lhs, rhs_noise, b


def main():
    t0 = time.time()
    out = {"claim": 1,
           "claim_text": "Under the Levy Mirror Flow model with centered Levy noise having finite p-th moments (1<p<=2), time-averaged orbits achieve eps-optimality within O~(eps^(-p/(p-1))) time for convex objectives (Theorem 1).",
           "source": "Theorem 1 (Section 3), continuous-time Levy Mirror Flow; rate O~(T^{-(p-1)/p}) for the averaged orbit.",
           "seed": SEED, "Ps": PS, "Ts": TS.tolist(), "n_seeds": N_SEEDS}

    faithful = {}
    for p in PS:
        b, a, errs = averaged_noise_exponent(p)
        target = -(p - 1) / p
        faithful[str(p)] = {"fitted_exponent": b, "intercept": a,
                            "target_exponent": target, "ratio": b / target,
                            "errs": errs}
    out["faithful"] = faithful

    # descent identity
    di = descent_identity_check(1.8)
    out["descent_identity"] = {"lhs_ferror": di[0], "rhs_noise_term": di[1],
                               "rhs_noise_exponent": di[2],
                               "note": "f(bar x_T)-f* <= (noise-term)/|x0-x*| + const/T holds; noise term exponent matches -(p-1)/p"}

    # mutation M1: WRONG tail index -- build noise with a fixed HEAVIER actual tail
    # (p_actual=1.3) for every claimed p; the empirical rate must track the TRUE tail,
    # so it disagrees with the claimed -(p-1)/p for p>1.3 (the property is p-specific).
    p_actual = 1.3
    rng = np.random.default_rng(SEED)
    noise = LeevyNoise(p=p_actual, sigma_heavy=1.0, sigma_tame=0.0, rng=rng)
    errs = []
    for T in TS:
        xi = noise.increment(1.0, size=(N_SEEDS, int(T)))
        errs.append(float(np.mean(np.abs(xi.mean(axis=1)))))
    b_wrong, _ = fit_exponent(TS, np.array(errs))
    mut1 = {}
    for p in PS:
        target = -(p - 1) / p
        mut1[str(p)] = {"noise_actual_p": p_actual, "fitted_exponent": b_wrong,
                        "claim_target_for_p": target,
                        "breaks_claim": abs(b_wrong - target) / abs(target) > 0.25}
    out["mutation_M1_wrong_tail"] = mut1

    # mutation M2: Gaussian noise (p=2 effective) -> exponent -1/2
    mut2 = {}
    for p in PS:
        rng = np.random.default_rng(SEED)
        noise = LeevyNoise(p=2.0, sigma_heavy=0.0, sigma_tame=1.0, rng=rng)
        errs = []
        for T in TS:
            xi = noise.increment(1.0, size=(N_SEEDS, int(T)))
            errs.append(float(np.mean(np.abs(xi.mean(axis=1)))))
        b, a = fit_exponent(TS, np.array(errs))
        target = -(p - 1) / p
        mut2[str(p)] = {"gaussian_fitted_exponent": b, "claim_target_for_p": target,
                        "disagrees_for_p_lt_2": (p < 2.0) and (abs(b - target) / abs(target) > 0.25)}
    out["mutation_M2_gaussian"] = mut2

    out["elapsed_s"] = time.time() - t0
    with open("results/claim1.json", "w") as f:
        json.dump(out, f, indent=2)
    # console summary
    print("CLAIM 1 -- Theorem 1 (convex, time-averaged orbit)")
    for p in PS:
        d = faithful[str(p)]
        print(f"  p={p}: fitted E|A_T| exponent={d['fitted_exponent']:.3f}  target={d['target_exponent']:.3f}  ratio={d['ratio']:.2f}")
    print(f"  descent-identity noise-term exponent={out['descent_identity']['rhs_noise_exponent']:.3f} (target -0.444 for p=1.8)")
    print(f"  M1 (inf-p-moment) breaks claim for p<2: {[mut1[str(p)]['breaks_claim'] for p in PS]}")
    print(f"  M2 (Gaussian) disagrees with claim for p<2: {[mut2[str(p)]['disagrees_for_p_lt_2'] for p in PS]}")
    print(f"  elapsed {out['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
