import sys, numpy as np, json
import verify_mdrcp as V

if __name__ == "__main__":
    tag = sys.argv[1]
    fn = V.run_clf if tag == "clf" else V.run_reg
    rows = []
    for s in range(10):
        r = fn(s); rows.append(r)
        print(tag, s, {k: [round(x, 4) for x in v] for k, v in r.items()}, flush=True)
    red = [100 * (r["AGG"][2] - r["MDCP"][2]) / r["AGG"][2] for r in rows]
    out = dict(reduction_mean=float(np.mean(red)), reduction_sd=float(np.std(red, ddof=1)),
               reduction_all=[round(x, 2) for x in red],
               mdcp_cov=float(np.mean([r["MDCP"][0] for r in rows])),
               mdcp_worst=float(np.mean([r["MDCP"][1] for r in rows])),
               agg_cov=float(np.mean([r["AGG"][0] for r in rows])),
               agg_worst=float(np.mean([r["AGG"][1] for r in rows])),
               src_cov=float(np.mean([r["SRC"][0] for r in rows])),
               src_worst=float(np.mean([r["SRC"][1] for r in rows])),
               size_mdcp=float(np.mean([r["MDCP"][2] for r in rows])),
               size_agg=float(np.mean([r["AGG"][2] for r in rows])))
    print("FINAL", json.dumps(out), flush=True)
    json.dump(out, open("res_%s.json" % tag, "w"), indent=1)
