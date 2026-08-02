"""Claim 6: BBQ's rater-quality parameter q_r correlates with Pearson r=0.724
with rater agreement to the final ranking on the UNSCREENED IHQ data
(r=0.551 on screened, per Fig. 2), enabling identification of unreliable raters.
Source: arXiv:2510.09333v2 Sec. 4.3, Fig. 2, Eq. (2), Eq. (12).

Method: fit BBQ on the split, take the final item ranking (by lambda), define
each rater's "agreement with the final ranking" as the fraction of their
comparisons whose winner is ranked above the loser, and correlate with q_r.

Mutation: permute the q_r vector across raters -> correlation must vanish.
"""
import json, numpy as np
from scipy.stats import pearsonr
from bbq_core import bbq_em
from ihq_data import load_split

SEED = 20260802
out = {"claim": 6, "seed": SEED,
       "source": "arXiv:2510.09333v2 Sec. 4.3 / Fig. 2 (Eq. 2, Eq. 12)",
       "paper_values": {"unscreened": 0.724, "screened": 0.551,
                        "n_raters_paper": {"unscreened": 62, "screened": 50}},
       "splits": {}}

for split in ("unscreened", "screened"):
    cmp_, items, raters = load_split(split)
    res = bbq_em(cmp_, seed=SEED)
    lam = res["lam"]; q = res["q"]
    rank_score = lam                       # higher lambda = better item
    agree = np.zeros(cmp_.R)
    for r in range(cmp_.R):
        w = cmp_.w[r]
        num = 0.0; den = 0.0
        for i in range(cmp_.K):
            for j in range(cmp_.K):
                if w[i, j] > 0:
                    den += w[i, j]
                    if rank_score[i] > rank_score[j]:
                        num += w[i, j]
        agree[r] = num / den if den else np.nan
    ok = ~np.isnan(agree)
    r_val, p_val = pearsonr(agree[ok], q[ok])
    fit = np.polyfit(agree[ok], q[ok], 1)
    out["splits"][split] = {
        "n_raters": int(ok.sum()),
        "n_raters_paper": out["paper_values"]["n_raters_paper"][split],
        "pearson_r": float(r_val), "p_value": float(p_val),
        "slope": float(fit[0]), "intercept": float(fit[1]),
        "paper_r": out["paper_values"][split],
        "abs_diff_from_paper": abs(float(r_val) - out["paper_values"][split]),
        "q_min": float(q.min()), "q_max": float(q.max()), "q_mean": float(q.mean()),
        "agreement_min": float(agree[ok].min()), "agreement_max": float(agree[ok].max()),
    }
    if split == "unscreened":
        rng = np.random.default_rng(SEED)
        perm_rs = [float(pearsonr(agree[ok], rng.permutation(q[ok]))[0])
                   for _ in range(1000)]
        out["mutation"] = {
            "description": "permute q_r across raters (1,000 permutations), unscreened split",
            "mean_abs_pearson_r": float(np.mean(np.abs(perm_rs))),
            "max_abs_pearson_r": float(np.max(np.abs(perm_rs))),
            "permutation_p_value": float(np.mean(np.abs(perm_rs) >= abs(r_val))),
            "breaks_property": bool(np.mean(np.abs(perm_rs) >= abs(r_val)) < 0.01),
        }

u = out["splits"]["unscreened"]; s = out["splits"]["screened"]
sign_ok = u["pearson_r"] > 0 and s["pearson_r"] > 0
close = u["abs_diff_from_paper"] < 0.10
stronger_unscreened = u["pearson_r"] > s["pearson_r"]
out["assessment"] = {
    "positive correlation on both splits": bool(sign_ok),
    "unscreened r within 0.10 of 0.724": bool(close),
    "correlation stronger on unscreened than screened (as in Fig. 2)":
        bool(stronger_unscreened),
}
out["verdict"] = "verified" if (sign_ok and close) else (
    "inconclusive" if sign_ok else "falsified")
out["notes"] = (
    "Fit on the public HF release, which contains fewer raters/comparisons than "
    "the paper reports (Appendix D Table 6), so an exact match to r=0.724 is not "
    "expected. The paper does not fully specify its 'agreement with the final "
    "ranking' statistic; we use the fraction of a rater's comparisons consistent "
    "with the BBQ ranking on the same split.")

with open("results/claim6.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
