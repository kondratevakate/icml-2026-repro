"""Reproduce the FunCQNet simulation *core claim* on CPU.

Claim under test (paper Sec. D.1, Table 3, tau = 0.5): a model that lets the
functional coefficients depend on the scalar covariates Z (Interaction, like
FunCQNet) has far lower RMSE for the coefficient functions than a reduced model
that ignores Z (No Interaction) -- in Scenarios S1/S2 where the true effect is
Z-modulated -- while the two are comparable in S3 where it is not. The paper
reports ~92% (alpha1) and ~75% (alpha2) RMSE reductions in S1 at tau=0.5.

We report the RMSE *ratio* No-Interaction / Interaction and the % reduction,
which is the scale-free, directional quantity the claim is about.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import dgp
import estimator as est


K_ALPHA = 8            # B-spline basis size for the coefficient functions
CENS_RATES = [0.0, 0.10, 0.30, 0.50]
SCENARIOS = ["S1", "S2", "S3"]


def rmse_alpha(true_fn, alpha_hat_rows, grid, Zrows):
    """sqrt( mean_Z int_t (ahat(t,Z) - atrue(t,Z))^2 dt )."""
    diff = alpha_hat_rows - true_fn          # (m, T)
    int_t = np.trapz(diff ** 2, grid, axis=1)
    return np.sqrt(int_t.mean())


def one_run(scenario, cens_rate, rng, n, n_grid):
    d = dgp.sample(scenario, n, cens_rate, rng, n_grid=n_grid)
    grid, w = d["grid"], est.trap_weights(d["grid"])
    Balpha = dgp.bspline_basis(grid, K_ALPHA)

    U1 = est.scores(d["X1"], Balpha, w)
    U2 = est.scores(d["X2"], Balpha, w)
    wt = est.ipcw_weights(d["Y"], d["delta"])
    y = d["logY"]

    # fit both models
    D0 = est.design_no_interaction(U1, U2)
    Di = est.design_interaction(U1, U2, d["Z"])
    c0 = est.wquantile_lp(D0, y, wt)
    ci = est.wquantile_lp(Di, y, wt)

    # evaluation Z set (fresh Monte-Carlo draw from the Z law)
    m = 400
    Z1 = rng.uniform(1, 2, m)
    Z2 = rng.integers(0, 2, m).astype(float)
    Z3 = rng.integers(0, 2, m).astype(float)
    Zev = np.column_stack([Z1, Z2, Z3])
    a1_true, a2_true = dgp.true_alpha(scenario, grid, Zev)

    # no-interaction: constant in Z -> broadcast to the m rows
    _, a1_0, a2_0 = est.alpha_from_no_interaction(c0, Balpha)
    a1_0 = np.tile(a1_0, (m, 1)); a2_0 = np.tile(a2_0, (m, 1))
    # interaction
    _, a1_i, a2_i = est.alpha_from_interaction(ci, Balpha, Zev)

    return dict(
        rmse1_int=rmse_alpha(a1_true, a1_i, grid, Zev),
        rmse2_int=rmse_alpha(a2_true, a2_i, grid, Zev),
        rmse1_noint=rmse_alpha(a1_true, a1_0, grid, Zev),
        rmse2_noint=rmse_alpha(a2_true, a2_0, grid, Zev),
        emp_cens=d["cens_rate"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--grid", type=int, default=60)
    ap.add_argument("--B", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20260723)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        t0 = time.time()
        rng = np.random.default_rng(args.seed)
        r = one_run("S1", 0.30, rng, args.n, args.grid)
        el = time.time() - t0
        total = len(SCENARIOS) * len(CENS_RATES) * args.B
        print(f"SMOKE S1 @30% cens: {r}")
        print(f"{el:.1f}s/run  ->  est. full run: {el*total/60:.1f} min "
              f"({total} runs)")
        assert r["rmse1_int"] < r["rmse1_noint"], "interaction should win in S1"
        print("smoke assertion passed: interaction < no-interaction in S1")
        return

    results = {}
    for sc in SCENARIOS:
        for cr in CENS_RATES:
            rows = []
            for b in range(args.B):
                rng = np.random.default_rng(args.seed + hash((sc, cr, b)) % 10 ** 6)
                rows.append(one_run(sc, cr, rng, args.n, args.grid))
            agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            agg["sd_rmse1_int"] = float(np.std([r["rmse1_int"] for r in rows]))
            agg["red1_pct"] = 100 * (1 - agg["rmse1_int"] / agg["rmse1_noint"])
            agg["red2_pct"] = 100 * (1 - agg["rmse2_int"] / agg["rmse2_noint"])
            results[f"{sc}@{int(cr*100)}"] = agg
            print(f"{sc} cens={int(cr*100):>2}%  "
                  f"RMSE1 int={agg['rmse1_int']:.3f} noint={agg['rmse1_noint']:.3f} "
                  f"(-{agg['red1_pct']:.0f}%)  "
                  f"RMSE2 int={agg['rmse2_int']:.3f} noint={agg['rmse2_noint']:.3f} "
                  f"(-{agg['red2_pct']:.0f}%)")
    json.dump(results, open("results.json", "w"), indent=2)
    print("\nsaved results.json")


if __name__ == "__main__":
    main()
