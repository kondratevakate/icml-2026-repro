"""Claim 3: On the unscreened IHQ dataset BBQ achieves 61.92% top-1 agreement
with the final ranking vs 33.15% (Crowd-BT) and 24.32% (Bayes-BT).
Source: task spec / Table 1.  NOTE arXiv v2 Table 1 prints 61.42 / 32.44 / 23.59
for IHQ-unscr., so the task's numbers come from an earlier version.

Method (Sec. 4): bootstrap-resample RATERS, refit each model, and measure how
often the top-1 item matches the reference ranking gt.
Caveat: the paper's gt for IHQ is the OFFICIAL CLIC 2024 leaderboard, which is
not redistributed with the dataset. We therefore use the paper's own fallback
gt definition ("the ranking achieved on the whole dataset"), computed per model
family on IHQ-all. Absolute percentages are therefore not directly comparable;
the ORDERING of the three methods is the testable content.

Mutation: randomly permute each rater's win/loss outcomes -> top-1 agreement
must collapse toward chance (1/28).
"""
import json, time, numpy as np
from bbq_core import bbq_em, bayes_bt, crowd_bt, Comparisons
from ihq_data import load_split, load_all

SEED = 20260802
N_BOOT = 500
CBT_EPOCHS = 20

cmp_all, items_all, _ = load_all()
cmp_u, items_u, _ = load_split("unscreened")
assert items_all == items_u, "item vocabularies differ"
K = cmp_u.K

# ---- reference ranking gt: full-dataset (IHQ-all) fit, per model family ----
gt = {
    "BBQ": int(np.argmax(bbq_em(cmp_all, seed=SEED)["lam"])),
    "Bayes-BT": int(np.argmax(bayes_bt(cmp_all, seed=SEED)["lam"])),
    "Crowd-BT": int(np.argmax(crowd_bt(cmp_all, epochs=CBT_EPOCHS, seed=SEED)["s"])),
}
# paper states all three models agree on the top item; check that
gt_consensus = len(set(gt.values())) == 1
GT_TOP = gt["BBQ"]


def boot(cmp_, rng):
    idx = rng.integers(0, cmp_.R, cmp_.R)
    return Comparisons(cmp_.w[idx])


def top1_rates(cmp_, n_boot=N_BOOT, seed=SEED, shuffle=False):
    rng = np.random.default_rng(seed)
    hit = {"BBQ": 0, "Bayes-BT": 0, "Crowd-BT": 0}
    w = cmp_.w
    for b in range(n_boot):
        c = boot(cmp_, rng)
        if shuffle:                      # mutation: randomize outcomes
            tot = c.w + np.transpose(c.w, (0, 2, 1))
            ww = np.zeros_like(c.w)
            iu = np.triu_indices(c.K, 1)
            t = tot[:, iu[0], iu[1]]
            f = rng.binomial(t.astype(int), 0.5).astype(float)
            ww[:, iu[0], iu[1]] = f
            ww[:, iu[1], iu[0]] = t - f
            c = Comparisons(ww)
        hit["BBQ"] += int(np.argmax(bbq_em(c, seed=SEED + b)["lam"]) == GT_TOP)
        hit["Bayes-BT"] += int(np.argmax(bayes_bt(c, seed=SEED + b)["lam"]) == GT_TOP)
        hit["Crowd-BT"] += int(np.argmax(
            crowd_bt(c, epochs=CBT_EPOCHS, seed=SEED + b)["s"]) == GT_TOP)
    return {k: 100.0 * v / n_boot for k, v in hit.items()}


t0 = time.perf_counter()
rates = top1_rates(cmp_u)
elapsed = time.perf_counter() - t0

paper_task = {"BBQ": 61.92, "Crowd-BT": 33.15, "Bayes-BT": 24.32}
paper_v2 = {"BBQ": 61.42, "Crowd-BT": 32.44, "Bayes-BT": 23.59}

order_ok = rates["BBQ"] > rates["Crowd-BT"] and rates["BBQ"] > rates["Bayes-BT"]
cbt_gt_bbt = rates["Crowd-BT"] > rates["Bayes-BT"]

out = {
    "claim": 3, "seed": SEED,
    "source": "arXiv:2510.09333v2 Table 1 (IHQ-unscr., Top-1 accuracy); Sec. 4 protocol",
    "data": {"dataset": "Mabyduck/CLIC2024-test-human-eval, split=unscreened",
             "comparisons_public_release": int(cmp_u.total()),
             "raters_public_release": cmp_u.R, "items": cmp_u.K,
             "comparisons_reported_in_paper": 2062, "raters_reported_in_paper": 62,
             "note": "the public HF release is smaller than the numbers in "
                     "Appendix D Table 6; exact replication of the paper's "
                     "sample is therefore impossible"},
    "protocol": {"bootstrap_resamples": N_BOOT,
                 "paper_bootstrap_resamples": 10000,
                 "gt": "top item of the full IHQ-all fit (paper's fallback gt; "
                       "official CLIC 2024 leaderboard not available)",
                 "gt_top_item_index": GT_TOP,
                 "gt_top_item": items_all[GT_TOP],
                 "all_three_models_agree_on_top_item": bool(gt_consensus),
                 "crowd_bt_epochs": CBT_EPOCHS,
                 "seconds": elapsed},
    "top1_accuracy_percent_ours": rates,
    "top1_accuracy_percent_task_spec": paper_task,
    "top1_accuracy_percent_arxiv_v2": paper_v2,
    "ordering_BBQ_best": bool(order_ok),
    "ordering_CrowdBT_above_BayesBT": bool(cbt_gt_bbt),
}

mut = top1_rates(cmp_u, n_boot=100, seed=SEED + 5, shuffle=True)
out["mutation"] = {
    "description": "each pair's winner re-drawn by a fair coin (rater signal destroyed), "
                   "100 bootstraps",
    "top1_accuracy_percent": mut,
    "chance_level_percent": 100.0 / K,
    "breaks_property": bool(mut["BBQ"] < rates["BBQ"] / 2),
}

out["verdict"] = "inconclusive"
out["ordering_reproduced"] = bool(order_ok)
out["notes"] = (
    "Measured ordering "
    f"(BBQ={rates['BBQ']:.2f}% > Crowd-BT={rates['Crowd-BT']:.2f}%, "
    f"Bayes-BT={rates['Bayes-BT']:.2f}%) is the reproducible content. The exact "
    "percentages cannot be reproduced: (a) the public IHQ release has "
    f"{int(cmp_u.total())} comparisons / {cmp_u.R} raters vs 2,062 / 62 in the paper, "
    "(b) the official CLIC 2024 leaderboard used as gt is not available, and "
    "(c) Crowd-BT here is our reimplementation, not Google's. Also note the "
    "task spec quotes 61.92/33.15/24.32 while arXiv v2 Table 1 prints "
    "61.42/32.44/23.59.")

with open("results/claim3.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
