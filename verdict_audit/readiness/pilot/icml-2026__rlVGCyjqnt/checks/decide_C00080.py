"""Merge per-seed result files of check_C00080.py and apply the pre-registered decision rule."""
import json, sys, glob
import numpy as np
sys.path.insert(0, "/work/checks")
sys.argv = [sys.argv[0]]
import check_C00080 as c

files = sorted(glob.glob("work/logs/C00080_results_seed*.json"))
res = {k: [] for k in c.PRESETS}; ri = []
for f in files:
    d = json.load(open(f))
    for name, r in d["runs"].items():
        res[name.rsplit("_seed", 1)[0]].append(r)
    ri += d.get("random_init", [])
print("files:", files)
print("random-init full-channel AUC:", [round(r["full"], 4) for r in ri])
for k in c.PRESETS:
    fm = [r["full"] for r in res[k]]; mm = [r["missing_mean"] for r in res[k]]
    print(f"{k:13s} n={len(fm)} full AUC {np.mean(fm):.4f} +- {np.std(fm, ddof=1):.4f} (per seed {[round(x,4) for x in fm]}) | "
          f"missing-channel mean AUC {np.mean(mm):.4f} +- {np.std(mm, ddof=1):.4f}")
    for sc in c.SCENARIOS:
        print(f"      {sc:16s} {np.mean([r[sc] for r in res[k]]):.4f}")
outcome, why, info = c.decide(res)
print("DECISION (pre-registered rule; primary metric = full-channel macro-AUC):", outcome, "-", why)
print(json.dumps(info))
miss = {k: float(np.mean([r["missing_mean"] for r in res[k]])) for k in c.PRESETS}
print("secondary: missing-channel mean AUC", {k: round(v, 4) for k, v in miss.items()})
json.dump({"outcome": outcome, "reason": why, "info": info, "missing_mean": miss}, open("work/logs/C00080_decision.json", "w"), indent=1)
