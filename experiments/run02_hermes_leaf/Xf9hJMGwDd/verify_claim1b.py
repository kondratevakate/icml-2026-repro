"""verify_claim1b.py -- companion sweep for Theorem 3.3.

Two questions left open by verify_claim1.py:
  (i)  does the ORACLE arm (Theorem 3.3's actual hypothesis, "Given nu_j and rho_j")
       stay valid across sample sizes and dimensions?  -- exhaustive over an (n,p) grid.
  (ii) is the inflation of the ESTIMATED arm a finite-sample effect that decays like the
       O_P(n^{-1/2}) of Theorem 4.1/4.2, and is the MUTATION arm strictly worse when the
       black box is allowed to overfit (large p/n)?

Usage: ./.venv/bin/python verify_claim1b.py
"""
import json, time
import numpy as np
from common import (gen_adjacent, oracle_nu, oracle_rho, ridge_fit, ridge_pred,
                    sko_pair_losses, hrt_style_losses, sign_test_pval)

GRID = [(100, 10), (200, 10), (400, 10), (800, 10), (1600, 10), (100, 40), (200, 40)]
N_SEEDS = 400
ALPHA = 0.10


def ridge_impute(A, b, lam=1e-2):
    return ridge_pred(ridge_fit(A, b, lam), A)


def one_config(n, p, n_seeds=N_SEEDS):
    pa, pb, pc = [], [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(hash((n, p, s)) % (2**31))
        X, y, beta, Sigma, sn = gen_adjacent(n, p, rng)
        nulls = np.where(beta == 0)[0]
        j = int(rng.choice(nulls))
        theta = ridge_fit(X, y)
        pred = lambda Z: ridge_pred(theta, Z)

        nu = oracle_nu(X, Sigma, j)
        rho = oracle_rho(X, y, Sigma, beta, sn, j)
        pa.append(sign_test_pval(*sko_pair_losses(X, y, j, nu, rho, pred, rng)))

        idx = np.delete(np.arange(p), j)
        nu_h = ridge_impute(X[:, idx], X[:, j])
        rho_h = ridge_impute(np.column_stack([X[:, idx], y]), X[:, j])
        pb.append(sign_test_pval(*sko_pair_losses(X, y, j, nu_h, rho_h, pred, rng)))

        pc.append(sign_test_pval(*hrt_style_losses(X, y, j, nu, pred, rng)))

    r = lambda v: float(np.mean(np.asarray(v) <= ALPHA))
    return {"n": n, "p": p, "oracle": r(pa), "estimated": r(pb), "mutation_asymmetric": r(pc)}


if __name__ == "__main__":
    t0 = time.time()
    rows = [one_config(n, p) for n, p in GRID]
    out = {
        "claim": "1 (companion sweep)",
        "source": "Theorem 3.3, Section 3.1; Theorem 4.2 (W1 control), Section 4.3",
        "command": "./.venv/bin/python verify_claim1b.py",
        "alpha": ALPHA, "n_seeds_per_config": N_SEEDS,
        "mc_standard_error": float(np.sqrt(ALPHA * (1 - ALPHA) / N_SEEDS)),
        "rows": rows,
        "wall_seconds": round(time.time() - t0, 2),
    }
    with open("results/claim1b.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"{'n':>6} {'p':>4} {'oracle':>8} {'estim':>8} {'mutation':>9}   (alpha={ALPHA}, mcse={out['mc_standard_error']:.4f})")
    for r in rows:
        print(f"{r['n']:>6} {r['p']:>4} {r['oracle']:>8.4f} {r['estimated']:>8.4f} {r['mutation_asymmetric']:>9.4f}")
    print("wall", out["wall_seconds"], "s")
