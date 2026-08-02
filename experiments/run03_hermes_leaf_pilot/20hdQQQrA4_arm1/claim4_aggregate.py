"""claim4_aggregate.py — combine results/claim4_<kind>_<seed>.json into results/claim4.json"""
import glob, json
import numpy as np

res = {"claim": 4, "seeds": [0, 1, 2, 3, 4],
       "paper_table2": {"NN_mse": 0.0045, "CAffNet_TF_mse": 0.0012,
                        "claimed_reduction_pct": 73.33},
       "command": "python claim4_worker.py <kind> <seed> 50000 ; python claim4_aggregate.py"}
for kind in ("nn_soft", "caffnet_tf", "tf_soft_mutation"):
    runs = [json.load(open(f)) for f in sorted(glob.glob(f"results/claim4_{kind}_*.json"))]
    if not runs:
        continue
    res[kind] = dict(runs=runs, n_seeds=len(runs), epochs=runs[0]["epochs"],
                     mse_mean=float(np.mean([r["mse"] for r in runs])),
                     mse_std=float(np.std([r["mse"] for r in runs])),
                     viol_max=float(max(r["viol_max"] for r in runs)),
                     viol_mean=float(np.mean([r["viol_mean"] for r in runs])),
                     viol_pct_mean=float(np.mean([r["viol_pct"] for r in runs])))
a, b = res["nn_soft"]["mse_mean"], res["caffnet_tf"]["mse_mean"]
res["measured_reduction_pct"] = float((a - b) / a * 100)
json.dump(res, open("results/claim4.json", "w"), indent=2)
print(json.dumps({k: v for k, v in res.items() if k not in ("paper_table2",)}, indent=2)[:2000])
