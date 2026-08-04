"""A4 (survival target): local accuracy of exact order-2 SurvSHAP-IQ.

Uses the CORRECTED n-Shapley construction (see report): Eq. (10) of the paper is
valid only for the TOP order |K| = k. For p=3, k=2, in Mobius terms m(.):
    phi^(2)_{i}  = m(i)  - m({1,2,3})/6      (first order)
    phi^(2)_{ij} = m(ij) + m({1,2,3})/2      (top order, == Eq. (10) SII)
This satisfies local accuracy exactly; naively applying Eq. (10) at both orders
does not.

REDUCED SETTINGS vs the paper (for runtime; stated explicitly in the report):
  - 5 timepoints instead of a dense grid over (0,70]
  - cumulative hazard integrated on a 97-point trapezoid grid
  - 3 seeds
Everything else follows the paper: n=1000, x~N(0,1), lambda=0.03, betas as given.
Paper target for ground-truth survival: sigma_bar < 0.001
"""
import numpy as np
from itertools import combinations
import audit_a4_localacc as base

P = 3
base.NGRID = 97
TIMES = np.array([14.0, 28.0, 42.0, 56.0, 70.0])


def mobius(v):
    out = {}
    for r in range(P + 1):
        for K in combinations(range(P), r):
            s = 0.0
            for lr in range(len(K) + 1):
                for L in combinations(K, lr):
                    s = s + (-1) ** (len(K) - lr) * v[tuple(sorted(L))]
            out[K] = s
    return out


def sigma_bar(G, kind, X):
    F = base.make_F(G, kind)
    sig = []
    for t in TIMES:
        v, b = base.value_all(F, t, X, X)
        mo = mobius(v)
        full = mo[(0, 1, 2)]
        Phi = sum(mo[(i,)] for i in range(P)) - full / 2.0
        Phi = Phi + sum(mo[K] for K in combinations(range(P), 2)) + 1.5 * full
        Fx = F(t, [X[:, i] for i in range(P)])
        sig.append(np.sqrt(np.mean((Fx - b - Phi) ** 2) / np.mean(Fx ** 2)))
    return float(np.mean(sig))


def main():
    print("=== A4 survival: sigma_bar, exact order-2 SurvSHAP-IQ (corrected n-Shapley) ===")
    print(f"    n=1000, |T|={len(TIMES)}, NGRID={base.NGRID}, paper target < 0.001\n")
    print("seed | " + " ".join(f"{s:>9}" for s in range(1, 11)) + " |      max   pass")
    for seed in range(3):
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, size=(1000, P))
        row = [sigma_bar(G, "survival", X) for G in base.scenarios().values()]
        mx = max(row)
        print(f"{seed:>4} | " + " ".join(f"{r:9.2e}" for r in row)
              + f" | {mx:9.2e} {str(mx < 1e-3):>6}", flush=True)


if __name__ == "__main__":
    main()
