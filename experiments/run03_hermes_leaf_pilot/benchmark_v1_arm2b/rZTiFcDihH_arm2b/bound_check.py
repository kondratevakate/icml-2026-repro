"""Post-hoc bound-satisfaction check: for every regret grid produced by the claim
scripts, compute R / sqrt(K*T). The anchored claims are UPPER bounds O~(sqrt(KT)),
so the primary test is that this ratio stays bounded (does not grow like sqrt(T)).
Writes the ratios back into results/claimN.json under 'bound_ratio'."""
import json, glob, numpy as np


def grids(d):
    out = {}
    if "per_s" in d:
        for s, v in d["per_s"].items():
            out[f"s{s}"] = v["regret_grid"]
    if "part_A_upper" in d:
        out["upper"] = d["part_A_upper"]["regret_grid"]
    if "part_B" in d:
        for s, v in d["part_B"].items():
            out[s] = v["regret_grid"]
    if "part_B_regret_grid" in d:
        out["main"] = d["part_B_regret_grid"]
    return out


for f in sorted(glob.glob("results/claim*.json")):
    d = json.load(open(f))
    br = {}
    for name, g in grids(d).items():
        r = {}
        for k, v in g.items():
            K = int(k.split("_")[0][1:]); T = int(k.split("_")[1][1:])
            r[k] = v / np.sqrt(K * T)
        br[name] = {"ratios": r, "max_ratio": float(max(r.values())),
                    "bound_holds_with_C1": bool(max(r.values()) <= 1.0)}
    if br:
        d["bound_ratio_R_over_sqrtKT"] = br
        json.dump(d, open(f, "w"), indent=2)
        print(f, {k: round(v["max_ratio"], 3) for k, v in br.items()})
