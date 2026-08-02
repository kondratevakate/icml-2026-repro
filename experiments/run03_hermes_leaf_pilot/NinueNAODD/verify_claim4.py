"""Claim 4: BBQ ranks FIRST in Kendall's Tau on 5 of 8 datasets and SECOND on
the remaining three, and achieves 100% top-1 agreement on MT-Bench, WD and HiFiC.
Source: arXiv:2510.09333v2 Table 1 (Sec. 3.3 / Sec. 4.1).

Two independent checks:
  A) INTERNAL AUDIT of the published Table 1: rank the three methods per dataset
     from the paper's own numbers and count firsts/seconds/thirds for BBQ.
     This needs no data and is exact.
  B) PARTIAL EMPIRICAL CHECK on the three IHQ splits we can actually obtain
     (Kendall's tau of each method's bootstrap ranking against the full-data
     reference ranking). The other five datasets (HUMAINE, MT-Bench, WD, HiFiC,
     ConHa) are not redistributed with the paper and are out of scope here.

Mutation (for the audit): perturb BBQ's tau on one dataset where it is first by
-0.01; the "first on 5 of 8" count must drop.
"""
import json, numpy as np
from bbq_core import bbq_em, bayes_bt, crowd_bt, Comparisons, kendall_tau
from ihq_data import load_split, load_all

SEED = 20260802
N_BOOT = 200
CBT_EPOCHS = 20

# ------------------------------------------------------- A) Table 1 audit ---
DATASETS = ["HUMAINE", "MT-Bench", "WD", "HiFiC", "ConHa",
            "IHQ-all", "IHQ-scr", "IHQ-unscr"]
TAU = {  # arXiv v2, Table 1 (bottom block)
    "Crowd-BT": [0.8933, 0.9743, 0.9279, 0.9366, 0.9180, 0.9210, 0.9238, 0.8375],
    "Bayes-BT": [0.9016, 0.9569, 0.9459, 0.9293, 0.9110, 0.9205, 0.9211, 0.8371],
    "BBQ":      [0.9025, 0.9675, 0.9359, 0.9525, 0.9265, 0.9211, 0.9204, 0.8431],
}
TOP1 = {  # Table 1 (top block)
    "Crowd-BT": [82.70, 100.00, 99.29, 98.63, 66.59, 85.46, 97.24, 32.44],
    "Bayes-BT": [97.40, 100.00, 100.00, 100.00, 57.30, 75.30, 98.89, 23.59],
    "BBQ":      [98.50, 100.00, 100.00, 100.00, 77.12, 99.34, 99.86, 61.42],
}


def bbq_positions(tau):
    pos = []
    for k in range(len(DATASETS)):
        vals = sorted([tau[m][k] for m in tau], reverse=True)
        pos.append(vals.index(tau["BBQ"][k]) + 1)
    return pos


pos = bbq_positions(TAU)
audit = {
    "bbq_kendall_rank_per_dataset": dict(zip(DATASETS, pos)),
    "n_first": int(sum(p == 1 for p in pos)),
    "n_second": int(sum(p == 2 for p in pos)),
    "n_third": int(sum(p == 3 for p in pos)),
    "claim_first_on_5": bool(sum(p == 1 for p in pos) == 5),
    "claim_second_on_remaining_3": bool(sum(p == 2 for p in pos) == 3),
    "datasets_where_bbq_is_third": [d for d, p in zip(DATASETS, pos) if p == 3],
}
top1_100 = {d: TOP1["BBQ"][DATASETS.index(d)] for d in ("MT-Bench", "WD", "HiFiC")}
audit["bbq_top1_on_MTBench_WD_HiFiC"] = top1_100
audit["claim_100pct_top1_on_those_three"] = bool(all(v == 100.0 for v in top1_100.values()))

# mutation of the audit
TAU_MUT = {m: list(v) for m, v in TAU.items()}
TAU_MUT["BBQ"][0] -= 0.01                       # HUMAINE: 0.9025 -> 0.8925
pos_mut = bbq_positions(TAU_MUT)
audit_mutation = {
    "description": "subtract 0.01 from BBQ's HUMAINE Kendall tau in Table 1",
    "n_first_after": int(sum(p == 1 for p in pos_mut)),
    "breaks_property": bool(sum(p == 1 for p in pos_mut) != 5),
}

# ------------------------------------------ B) empirical check on IHQ only ---
cmp_all, items, _ = load_all()
ref = {"BBQ": np.log(bbq_em(cmp_all, seed=SEED)["lam"]),
       "Bayes-BT": np.log(bayes_bt(cmp_all, seed=SEED)["lam"]),
       "Crowd-BT": crowd_bt(cmp_all, epochs=CBT_EPOCHS, seed=SEED)["s"]}
REF = ref["BBQ"]     # single shared reference ranking

empirical = {}
for split_name, loader in [("IHQ-all", lambda: (cmp_all, items, None)),
                           ("IHQ-scr", lambda: load_split("screened")),
                           ("IHQ-unscr", lambda: load_split("unscreened"))]:
    cmp_, _, _ = loader()
    rng = np.random.default_rng(SEED)
    taus = {"BBQ": [], "Bayes-BT": [], "Crowd-BT": []}
    for b in range(N_BOOT):
        c = Comparisons(cmp_.w[rng.integers(0, cmp_.R, cmp_.R)])
        taus["BBQ"].append(kendall_tau(np.log(bbq_em(c, seed=SEED + b)["lam"]), REF))
        taus["Bayes-BT"].append(kendall_tau(np.log(bayes_bt(c, seed=SEED + b)["lam"]), REF))
        taus["Crowd-BT"].append(kendall_tau(crowd_bt(c, epochs=CBT_EPOCHS,
                                                     seed=SEED + b)["s"], REF))
    empirical[split_name] = {m: float(np.mean(v)) for m, v in taus.items()}
    best = max(empirical[split_name], key=empirical[split_name].get)
    empirical[split_name]["best"] = best

out = {
    "claim": 4, "seed": SEED,
    "source": "arXiv:2510.09333v2 Table 1 (+ Sec. 4.1 text)",
    "A_table1_audit": audit,
    "A_mutation": audit_mutation,
    "B_empirical_IHQ_only": {
        "protocol": f"{N_BOOT} rater-bootstraps per split, tau vs BBQ full-data "
                    f"ranking on IHQ-all; Crowd-BT is our reimplementation",
        "mean_kendall_tau": empirical,
        "caveat": "reference ranking is BBQ's own full-data fit, which favours BBQ; "
                  "treat as a consistency check, not as evidence for the claim",
    },
    "verdict": "falsified",
    "notes": (
        "Sub-claim '100% top-1 on MT-Bench, WD, HiFiC' is CONFIRMED by Table 1 "
        "(but not independently reproduced -- those datasets are not available here). "
        "Sub-claim 'first on 5 of 8' is CONFIRMED by Table 1. "
        "Sub-claim 'second on the remaining three' is FALSE against the paper's own "
        f"Table 1: BBQ is second on {audit['n_second']} datasets and THIRD on "
        f"{audit['datasets_where_bbq_is_third']} (BBQ 0.9204 < Bayes-BT 0.9211 < "
        "Crowd-BT 0.9238 on IHQ-screened). The conjunctive claim is therefore false "
        "as stated. Independent dataset-level reproduction was not possible for "
        "5 of 8 datasets."),
}

with open("results/claim4.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
