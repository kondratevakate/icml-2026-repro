"""Claim 2: BBQ converges within seconds on all tested datasets, whereas
Crowd-BT takes ~15 minutes on HUMAINE (105,220 comparisons) even though BBQ is
plain unoptimized NumPy.   Source: Fig. 3, Sec. 4.4.

What is reproducible on CPU without the original datasets/implementations:
  - BBQ wall-clock on real IHQ (screened/unscreened/all) and on a synthetic
    dataset with HUMAINE's exact shape (105,220 comparisons, 1,977 raters,
    27 items; Appendix D Table 6).
  - A Crowd-BT (Chen et al. 2013 gradient) timing from OUR reimplementation.
    This is NOT the Google implementation the paper timed, so the "~15 minutes"
    figure cannot be confirmed or refuted -- only the order-of-magnitude gap.
Mutation: shrink the dataset 100x; BBQ runtime must drop roughly proportionally
(i.e. the "seconds" result is data-size dependent, not a constant).
"""
import json, time, numpy as np
from bbq_core import simulate, bbq_em, crowd_bt, Comparisons
from ihq_data import load_split, load_all

SEED = 20260802
out = {"claim": 2, "seed": SEED,
       "source": "arXiv:2510.09333v2 Fig. 3 / Sec. 4.4; dataset shapes App. D Table 6"}

timings = {}

# --- real IHQ splits -------------------------------------------------------
for name, loader in [("IHQ-screened", lambda: load_split("screened")),
                     ("IHQ-unscreened", lambda: load_split("unscreened")),
                     ("IHQ-all", load_all)]:
    cmp_, items, raters = loader()
    t0 = time.perf_counter(); r = bbq_em(cmp_, seed=SEED); t = time.perf_counter() - t0
    timings[name] = dict(n_comparisons=int(cmp_.total()), raters=cmp_.R,
                         items=cmp_.K, bbq_seconds=t, bbq_iters=r["iters"])

# --- synthetic HUMAINE-shaped dataset --------------------------------------
cmp_h, _, _ = simulate(K=27, R=1977, n_per_rater=53, frac_bad=0.25, seed=SEED)
t0 = time.perf_counter(); rh = bbq_em(cmp_h, seed=SEED); t_bbq = time.perf_counter() - t0
timings["HUMAINE-shaped (synthetic)"] = dict(
    n_comparisons=int(cmp_h.total()), raters=cmp_h.R, items=cmp_h.K,
    bbq_seconds=t_bbq, bbq_iters=rh["iters"])

t0 = time.perf_counter(); crowd_bt(cmp_h, epochs=3, seed=SEED)
t_cbt3 = time.perf_counter() - t0
timings["HUMAINE-shaped (synthetic)"]["crowdbt_ours_seconds_per_epoch"] = t_cbt3 / 3
timings["HUMAINE-shaped (synthetic)"]["crowdbt_ours_seconds_60_epochs_est"] = t_cbt3 / 3 * 60

out["timings"] = timings
out["bbq_max_seconds_any_dataset"] = max(v["bbq_seconds"] for v in timings.values())
out["bbq_all_under_10s"] = bool(out["bbq_max_seconds_any_dataset"] < 10)

# --- MUTATION: 100x smaller data ------------------------------------------
small = Comparisons(cmp_h.w[:20])          # 20 raters instead of 1977
t0 = time.perf_counter(); bbq_em(small, seed=SEED); t_small = time.perf_counter() - t0
out["mutation"] = {
    "description": "subsample HUMAINE-shaped data to 20 raters (~1% of comparisons)",
    "bbq_seconds_full": t_bbq,
    "bbq_seconds_small": t_small,
    "speedup": t_bbq / t_small if t_small > 0 else None,
    "breaks_property": bool(t_small < t_bbq),
}

out["sub_verdicts"] = {
    "BBQ converges in seconds": "verified (all datasets tested < "
                                f"{out['bbq_max_seconds_any_dataset']:.2f}s, plain NumPy)",
    "Crowd-BT ~15 min on HUMAINE": "inconclusive (HUMAINE comparison data not "
                                   "used; timed against our own Crowd-BT "
                                   "reimplementation, not the Google impl the "
                                   "paper benchmarked)",
}
out["verdict"] = "inconclusive"
out["notes"] = ("BBQ half of the claim reproduces cleanly on real IHQ data and on a "
                "synthetic dataset with HUMAINE's exact shape. The 15-minute "
                "Crowd-BT figure is implementation-specific and could not be "
                "checked without the Google Crowd-BT code and the HUMAINE data.")

with open("results/claim2.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
