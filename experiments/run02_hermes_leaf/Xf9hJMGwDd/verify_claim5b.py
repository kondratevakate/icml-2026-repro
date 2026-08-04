"""verify_claim5b.py -- harder regime for Figures 4/5 (Section 5.1).

verify_claim5.py saturated: at n=200 with a strong linear signal every method reached
power 1.0, so the anchored comparison "power(SKO) > power(HRT)" was untestable
(ceiling effect). This script lowers n and raises the noise so that power lies strictly
inside (0,1) and the split-vs-no-split difference can actually be measured.

Still `toy`: the black box is ridge, not the paper's NN / GB / RF.

Usage: ./.venv/bin/python verify_claim5b.py
"""
import json, time
import numpy as np
import verify_claim5 as V
from common import gen_adjacent

V.K_HRT = 100
V.ALPHA = 0.05
N_SEEDS = 250
CONFIGS = [(60, 12, 6.0), (80, 12, 4.0), (120, 12, 4.0)]


def run(n, p, noise, n_seeds=N_SEEDS):
    acc = {m: {"t1": [], "pow": []} for m in ["SKO-1", "SKO-5", "HRT"]}
    for s in range(n_seeds):
        rng = np.random.default_rng(300_000 + s)
        X, y, beta, Sig, sn = gen_adjacent(n, p, rng, sigma_noise=noise)
        nn = beta != 0
        res = {
            "SKO-1": V.sko_pvals(X, y, Sig, beta, sn, 1, rng, False),
            "SKO-5": V.sko_pvals(X, y, Sig, beta, sn, 5, rng, False),
            "HRT":   V.hrt_pvals(X, y, Sig, beta, sn, rng, False),
        }
        for m, pv in res.items():
            acc[m]["t1"].append(np.mean(pv[~nn] <= V.ALPHA))
            acc[m]["pow"].append(np.mean(pv[nn] <= V.ALPHA))
    return {m: {"type_I_error": float(np.mean(v["t1"])),
                "type_I_mcse": float(np.std(v["t1"]) / np.sqrt(n_seeds)),
                "power": float(np.mean(v["pow"])),
                "power_mcse": float(np.std(v["pow"]) / np.sqrt(n_seeds))}
            for m, v in acc.items()}


if __name__ == "__main__":
    t0 = time.time()
    rows = {}
    for n, p, noise in CONFIGS:
        rows[f"n={n},p={p},noise={noise}"] = run(n, p, noise)
    out = {
        "claim": "5 (harder regime, unsaturated)",
        "source": "Figures 4 and 5, Section 5.1",
        "command": "./.venv/bin/python verify_claim5b.py",
        "VERDICT_CAP": "toy -- ridge black box, not the paper's NN/GB/RF",
        "config": {"alpha": V.ALPHA, "n_seeds": N_SEEDS, "K_HRT": V.K_HRT,
                   "nu,rho": "ORACLE closed form (isolates the split effect from imputer error)"},
        "results": rows,
        "wall_seconds": round(time.time() - t0, 2),
    }
    with open("results/claim5b.json", "w") as f:
        json.dump(out, f, indent=2)
    for k, v in rows.items():
        print(k)
        for m, r in v.items():
            print(f"   {m:6s} typeI={r['type_I_error']:.4f}+-{r['type_I_mcse']:.4f}  "
                  f"power={r['power']:.4f}+-{r['power_mcse']:.4f}")
    print("wall", out["wall_seconds"])
