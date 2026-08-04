"""verify_claim5.py -- Corollary 1 (discrete, strongly convex, iteration complexity).

Claim: For strongly convex functions, the discrete-time method requires
Omega((sigma^p / eps)^{1/(p-1)} log(1/eps)) iterations to reach eps-accuracy (Corollary 1).

Reproduction (first principles): on mu-strongly-convex f=1/2 mu x^2, run discrete SDA with
i.i.d. Levy gradient noise. Measure the iteration count N(eps) to first reach
f(x_k)-f* < eps, over many seeds, for varying eps and sigma_heavy. Fit the exponents
    N(eps) ~ eps^{-a},   N ~ sigma_heavy^{b}
and compare a to 1/(p-1) and b to p/(p-1).

Honest note: at feasible scale the MEDIAN first-hit time is dominated by the deterministic
geometric decay of the strongly-convex drift (apparent exponent ~0.5-1, roughly logarithmic),
not the claimed polynomial Omega((sigma^p/eps)^{1/(p-1)}); the polynomial is a worst-case /
lower-bound statement whose heavy-tail regime is not reached by the median at this scale. This
is reported transparently; the heavy-tail (90th-percentile / larger p) direction is checked.
"""
import json, time
import numpy as np
from levy_core import LeevyNoise, make_quad, sda_discrete, fit_exponent

SEED = 12345
PS = [1.5, 2.0]
EPSS = np.array([0.002, 0.005, 0.01, 0.02, 0.05])
SIGS = np.array([0.5, 1.0, 2.0])
ETA = 0.02
X0 = 20.0
NMAX = 20000
SEEDS = 30


def iter_to_eps(p, eps, sh, st, eta, x0, seed, q=50, Nmax=NMAX, seeds=SEEDS, noise_dt=1.0):
    obj = make_quad(1.0)
    rng = np.random.default_rng(seed)
    noise = LeevyNoise(p=p, sigma_heavy=sh, sigma_tame=st, rng=rng)
    xs, _ = sda_discrete(obj, noise, Nmax, eta, x0, seeds, seed, average='last', noise_dt=noise_dt)
    err = 0.5 * xs ** 2
    hit = (err < eps).argmax(axis=1)
    never = ~(err < eps).any(axis=1)
    hit[never] = Nmax
    return float(np.percentile(hit, q))


def main():
    t0 = time.time()
    out = {"claim": 5,
           "claim_text": "For strongly convex functions, the discrete-time method requires Omega((sigma^p/eps)^{1/(p-1)} log(1/eps)) iterations to reach eps-accuracy (Corollary 1).",
           "source": "Corollary 1 (Section 5), iteration-complexity lower bound.",
           "seed": SEED, "Ps": PS, "eps": EPSS.tolist(), "sigma": SIGS.tolist(), "eta": ETA, "x0": X0, "Nmax": NMAX, "n_seeds": SEEDS}

    faithful = {}
    for p in PS:
        Ns_med = np.array([iter_to_eps(p, e, 1.0, 0.0, ETA, X0, SEED, q=50) for e in EPSS])
        Ns_p90 = np.array([iter_to_eps(p, e, 1.0, 0.0, ETA, X0, SEED, q=90) for e in EPSS])
        a_med, _ = fit_exponent(1.0 / EPSS, Ns_med)
        a_p90, _ = fit_exponent(1.0 / EPSS, Ns_p90)
        Ns_sig = np.array([iter_to_eps(p, 0.01, s, 0.0, ETA, X0, SEED, q=90) for s in SIGS])
        b_sig, _ = fit_exponent(SIGS, Ns_sig)
        faithful[str(p)] = {"N_eps_median": Ns_med.tolist(), "eps_exponent_median": a_med,
                            "N_eps_p90": Ns_p90.tolist(), "eps_exponent_p90": a_p90,
                            "target_eps_exponent": 1.0 / (p - 1),
                            "sigma_exponent_p90": b_sig, "target_sigma_exponent": p / (p - 1)}
    out["faithful"] = faithful

    # mutation: heavy vs gaussian -- does the tail make the worst-case steeper?
    mut = {}
    for p in PS:
        Ns_h = np.array([iter_to_eps(p, e, 1.0, 0.0, ETA, X0, SEED, q=90) for e in EPSS])
        Ns_g = np.array([iter_to_eps(2.0, e, 0.0, 1.0, ETA, X0, SEED, q=90) for e in EPSS])
        bh, _ = fit_exponent(1.0 / EPSS, Ns_h); bg, _ = fit_exponent(1.0 / EPSS, Ns_g)
        mut[str(p)] = {"eps_exponent_p90_heavy": bh, "eps_exponent_p90_gaussian": bg,
                       "tail_shifts_exponent": abs(bh - bg) > 0.1}
    out["mutation_heavy_vs_gaussian"] = mut

    out["elapsed_s"] = time.time() - t0
    with open("results/claim5.json", "w") as f:
        json.dump(out, f, indent=2)
    print("CLAIM 5 -- Corollary 1 (iteration complexity)")
    for p in PS:
        d = faithful[str(p)]
        print(f"  p={p}: eps-exp median={d['eps_exponent_median']:.3f}  p90={d['eps_exponent_p90']:.3f}  target={d['target_eps_exponent']:.3f}")
        print(f"        sigma-exp p90={d['sigma_exponent_p90']:.3f}  target={d['target_sigma_exponent']:.3f}")
        print(f"        N_eps_median={np.round(d['N_eps_median'],0)}")
    print(f"  elapsed {out['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
