import numpy as np, json
from verify_mdrcp import run_clf, run_reg

if __name__ == "__main__":
    out = {}
    for tag, fn in (("clf", run_clf), ("reg", run_reg)):
        rows = [fn(s) for s in range(12)]
        red = [100 * (r["AGG"][2] - r["MDCP"][2]) / r["AGG"][2] for r in rows]
        out[tag] = dict(
            reduction_mean=float(np.mean(red)), reduction_sd=float(np.std(red, ddof=1)),
            reduction_all=[round(x, 2) for x in red],
            mdcp_cov=float(np.mean([r["MDCP"][0] for r in rows])),
            mdcp_worst=float(np.mean([r["MDCP"][1] for r in rows])),
            agg_cov=float(np.mean([r["AGG"][0] for r in rows])),
            agg_worst=float(np.mean([r["AGG"][1] for r in rows])),
            src_cov=float(np.mean([r["SRC"][0] for r in rows])),
            src_worst=float(np.mean([r["SRC"][1] for r in rows])),
            size_mdcp=float(np.mean([r["MDCP"][2] for r in rows])),
            size_agg=float(np.mean([r["AGG"][2] for r in rows])))
        print(tag, json.dumps(out[tag], indent=1))
    json.dump(out, open("results.json", "w"), indent=1)
