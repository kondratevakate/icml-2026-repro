"""verify_claim3.py -- Theorem 3 (strongly convex, first-passage / hitting time).

Claim: The first-passage (hitting) time to within delta of the optimum for strongly
convex objectives is O~(delta^{-2p/(p-1)}) (Theorem 3).

Reproduction (first principles): On mu-strongly-convex f=1/2 mu x^2, run discrete SDA with
step eta and Levy gradient noise. Measure the mean first-passage time T(delta) = first
iteration k with |x_k| < delta, for delta below the stationary uncertainty-ball radius
(delta in a regime where a genuine downward fluctuation is required). Fit the exponent of
T(delta) ~ delta^{exponent} and compare to the claimed -2p/(p-1).

Honest note: at feasible sample scale the empirical first-passage of a STATIONARY
heavy-tailed process to a small target is governed by a geometric waiting time
(~ delta^{-1}), shallower than the claimed O~(delta^{-2p/(p-1)}); the steeper claimed
polynomial is a worst-case / asymptotic statement (e.g. fresh-start or averaged-orbit
hitting time) not reproduced by the stationary-iterate first-passage here. This is reported
transparently.
"""
import json, time
import numpy as np
from levy_core import LeevyNoise, make_quad, sda_discrete, fit_exponent

SEED = 12345
PS = [1.5, 2.0]
DELTAS = np.array([0.02, 0.035, 0.06, 0.1])
ETA = 0.05
NMAX = 30000
SEEDS = 25


def fp_discrete(p, delta, sh, st, eta, seed, Nmax=NMAX, seeds=SEEDS, noise_dt=1.0):
    obj = make_quad(1.0)
    rng = np.random.default_rng(seed)
    noise = LeevyNoise(p=p, sigma_heavy=sh, sigma_tame=st, rng=rng)
    xs, _ = sda_discrete(obj, noise, Nmax, eta, 1.0, seeds, seed, average='last', noise_dt=noise_dt)
    dist = np.abs(xs)
    hit = (dist < delta).argmax(axis=1)
    never = ~(dist < delta).any(axis=1)
    hit[never] = Nmax
    return float(hit.mean())


def main():
    t0 = time.time()
    out = {"claim": 3,
           "claim_text": "The first-passage (hitting) time to within delta of the optimum for strongly convex objectives is O~(delta^{-2p/(p-1)}) (Theorem 3).",
           "source": "Theorem 3 (Section 3/4), first-passage time scaling.",
           "seed": SEED, "Ps": PS, "deltas": DELTAS.tolist(), "eta": ETA, "Nmax": NMAX, "n_seeds": SEEDS}
    faithful = {}
    for p in PS:
        # heavy (tame off)
        Ts_h = np.array([fp_discrete(p, d, 1.0, 0.0, ETA, SEED) for d in DELTAS])
        b_h, _ = fit_exponent(DELTAS, Ts_h)
        # gaussian (tame only, p=2 effective)
        Ts_g = np.array([fp_discrete(2.0, d, 0.0, 1.0, ETA, SEED) for d in DELTAS])
        b_g, _ = fit_exponent(DELTAS, Ts_g)
        faithful[str(p)] = {"T_delta_heavy": Ts_h.tolist(), "exponent_heavy": b_h,
                            "target": -2 * p / (p - 1),
                            "T_delta_gaussian": Ts_g.tolist(), "exponent_gaussian": b_g}
    out["faithful"] = faithful

    # mutation: heavy vs gaussian exponent shift with p
    mut = {}
    for p in PS:
        mut[str(p)] = {"exponent_heavy": faithful[str(p)]["exponent_heavy"],
                       "exponent_gaussian": faithful[str(p)]["exponent_gaussian"],
                       "tail_shifts_exponent": abs(faithful[str(p)]["exponent_heavy"] - faithful[str(p)]["exponent_gaussian"]) > 0.1}
    out["mutation_heavy_vs_gaussian"] = mut

    out["elapsed_s"] = time.time() - t0
    with open("results/claim3.json", "w") as f:
        json.dump(out, f, indent=2)
    print("CLAIM 3 -- Theorem 3 (first-passage time)")
    for p in PS:
        d = faithful[str(p)]
        print(f"  p={p}: exponent_heavy={d['exponent_heavy']:.3f}  exponent_gaussian={d['exponent_gaussian']:.3f}  target={d['target']:.3f}")
        print(f"        T_delta_heavy={np.round(d['T_delta_heavy'],1)}")
    print(f"  elapsed {out['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
