"""A3: Theorem 3.3 (asymmetric propagation), using the PAPER'S OWN constructed example.

Verbatim from paper.txt (Appendix A.2, proof of Theorem 3.3):
  x = (x1,x2,x3), X1,X2,X3 iid ~ N(0,1)
  G(t|x) = x1^2 + x2 + x3 + x1 * x2^2 * t
  Z = {1,2} (time-dependent interaction g_{12} = x1 x2^2 t)
  Iid = {{1},{2},{3}}  (g1 = x1^2, g2 = x2, g3 = x3, all time-independent)
Paper's analytic result (A27-A29):
  f_{1}(t|x) = x1^2 + t*x1 - 1        <-- DOWNWARD propagation: depends on t
Theorem 3.3 part 2: for any L superset of Z with L in Iid, L stays time-independent
  -> L = {1,2,3} must have NO t-dependence  (NO UPWARD propagation)

Indices here are 0-based: Z = (0,1), upward set = (0,1,2).
"""
import numpy as np
from itertools import combinations

P = 3
TIMES = np.array([0.5, 1.0, 2.0, 4.0])
LOGH0 = np.log(0.1)  # any constant; cancels in pure effects


def G(t, x):
    return x[0] ** 2 + x[1] + x[2] + x[0] * x[1] ** 2 * t


def logh(t, x):
    return LOGH0 + G(t, x)


ALL = [S for r in range(1, P + 1) for S in combinations(range(P), r)]


def _cols(X, M, xr):
    return [xr[i] if i in M else X[:, i] for i in range(P)]


def pure_effects(t, xr, X):
    f = {(): float(np.mean(logh(t, [X[:, i] for i in range(P)])))}
    for M in sorted(ALL, key=len):
        f[M] = float(np.mean(logh(t, _cols(X, M, xr)))) - sum(
            f[Z] for Z in f if set(Z) < set(M))
    return f


def run(seed, n_ref, k_tol=6.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_ref, P))
    xr = np.array([0.8, -0.4, 1.3])
    vals = {t: pure_effects(t, xr, X) for t in TIMES}
    se = float(np.std(logh(TIMES[0], [X[:, i] for i in range(P)])) / np.sqrt(n_ref))
    tol = max(k_tol * se, 1e-4)
    spread = {M: float(max(vals[t][M] for t in TIMES) - min(vals[t][M] for t in TIMES))
              for M in ALL}
    td = {M: spread[M] > tol for M in ALL}
    # analytic check of the paper's closed form f_{1}(t|x) = x1^2 + t*x1 - 1
    analytic_err = max(abs(vals[t][(0,)] - (xr[0] ** 2 + t * xr[0] - 1)) for t in TIMES)
    return dict(tol=tol, spread=spread, td=td, analytic_err=analytic_err,
                down=td[(0,)], up_TI=not td[(0, 1, 2)], Z_td=td[(0, 1)])


def main():
    N = 200000
    print("=== A3: Theorem 3.3, paper's own example  G = x1^2+x2+x3+x1*x2^2*t ===")
    print(f"    Z={{x1,x2}}, N_ref={N}, tol=6*SE, times={list(TIMES)}")
    print(f"{'seed':>4} {'Z=TD':>6} {'DOWN f1=TD':>11} {'UP f123=TI':>11} "
          f"{'|f1 - analytic|':>16} {'tol':>9}")
    ok = 0
    for s in range(10):
        r = run(s, N)
        good = r["Z_td"] and r["down"] and r["up_TI"]
        ok += good
        print(f"{s:>4} {str(r['Z_td']):>6} {str(r['down']):>11} {str(r['up_TI']):>11} "
              f"{r['analytic_err']:>16.2e} {r['tol']:>9.2e}")
    print(f"\nTheorem 3.3 fully satisfied on {ok}/10 seeds")
    r = run(0, N)
    print("\nseed 0 spreads over t (max-min):")
    for M in ALL:
        print(f"  {M}: {r['spread'][M]:.3e}   -> {'TD' if r['td'][M] else 'TI'}")


if __name__ == "__main__":
    main()
