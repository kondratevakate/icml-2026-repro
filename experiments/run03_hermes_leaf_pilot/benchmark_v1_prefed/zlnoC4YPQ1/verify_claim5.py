"""verify_claim5.py — Claim 5: on the corrupted Forrester benchmark the RCGP-based methods
(FC-RCGP-UCB, A2-RCGP-UCB) attain lower cumulative regret than GP-UCB, Student-t Process UCB and
DiagnosticsGP, at a corruption budget T_c = O(T^{1/3}) (Sec 5.3, Figure 1; config Appendix I.4.2).

The paper reports figures only (no numeric regret values), so the reproducible content is the
ORDERING of the five methods' mean cumulative regret over seeds. Two settings are run because the
main text and the appendix specify different adversary parameters:
  A (Sec 5.3 text):      T = 100 iterations, near = 0.2, far = 0.5
  B (Appendix I.4.2):    T = 30  iterations, near = 0.1, far = 0.4
both with low = -10.0, high = 25.0, T_c = ceil(T^{1/3}), 5 uncorrupted initial points, 10 seeds,
sigma_noise^2 = 1. The uncorrupted setting (Fig 3) is run as well.
RCGP variants: 'est' = T_c estimated as #points outside the plateau; 'zero' = T_c set to 0 inside
beta_t (both are endorsed in Appendix I.1).
MUTATION: disable the robustness mechanism by making the plateau infinite (L = 1e9), which turns
the RCGP exactly into a GP -> the RCGP curves must collapse onto GP-UCB.
"""
import json, numpy as np
from bo import run_bo

SEEDS = list(range(10))
BASELINES = ["gp_ucb", "studentt_ucb", "diagnostics_gp"]
SETTINGS = {
    "A_maintext_T100": dict(n_iter=100, adv_kw=dict(near=0.2, far=0.5, low=-10.0, high=25.0)),
    "B_appendix_T30": dict(n_iter=30, adv_kw=dict(near=0.1, far=0.4, low=-10.0, high=25.0)),
}


def agg(vals):
    v = np.array(vals, float)
    return {"mean": float(v.mean()), "sem": float(v.std(ddof=1) / np.sqrt(len(v))),
            "median": float(np.median(v)), "per_seed": [float(x) for x in v]}


out = {"claim": 5, "source": "Section 5.3 / Figure 1 (+ Appendix I.4.2 configuration)",
       "command": ".venv/bin/python verify_claim5.py",
       "note": "paper reports no numeric regret values; the testable content is the method ordering"}

for name, cfg in SETTINGS.items():
    n_iter = cfg["n_iter"]
    res = {"n_iter": n_iter, "T_c_budget": int(np.ceil(n_iter ** (1 / 3))), **cfg["adv_kw"]}
    runs = {}
    for m in BASELINES:
        runs[m] = agg([run_bo(m, s, n_iter, corrupt=True, adv_kw=cfg["adv_kw"])["cum_regret"]
                       for s in SEEDS])
    # L variants: "q95" = 95%-quantile heuristic, "1.96" = the fixed value recommended in App I.1
    for m in ("fc_rcgp_ucb", "a2_rcgp_ucb"):
        for mode in ("est", "zero"):
            for lname, lval in (("q95", None), ("1.96", 1.96)):
                runs[f"{m}[Tc={mode},L={lname}]"] = agg(
                    [run_bo(m, s, n_iter, corrupt=True, tc_mode=mode, fixed_L=lval,
                            adv_kw=cfg["adv_kw"])["cum_regret"] for s in SEEDS])
    # MUTATION: L -> infinity turns RCGP into a plain GP
    for m in ("fc_rcgp_ucb", "a2_rcgp_ucb"):
        runs[f"MUT_{m}[L=inf]"] = agg(
            [run_bo(m, s, n_iter, corrupt=True, fixed_L=1e9, tc_mode="zero",
                    adv_kw=cfg["adv_kw"])["cum_regret"] for s in SEEDS])
    res["cum_regret"] = runs
    order = sorted(runs, key=lambda k: runs[k]["mean"])
    res["ranking_by_mean"] = [(k, round(runs[k]["mean"], 3)) for k in order]
    rcgp_keys = [k for k in runs if k.startswith(("fc_", "a2_"))]
    worst_rcgp = max(runs[k]["mean"] for k in rcgp_keys)
    best_rcgp = min(runs[k]["mean"] for k in rcgp_keys)
    res["best_rcgp_variant"] = min(rcgp_keys, key=lambda k: runs[k]["mean"])
    best_baseline = min(runs[b]["mean"] for b in BASELINES)
    res["best_rcgp_mean"] = best_rcgp
    res["worst_rcgp_mean"] = worst_rcgp
    res["best_baseline_mean"] = best_baseline
    res["all_rcgp_variants_beat_all_baselines"] = bool(worst_rcgp < best_baseline)
    res["some_rcgp_variant_beats_all_baselines"] = bool(best_rcgp < best_baseline)
    res["gp_ucb_is_worst_baseline"] = bool(
        runs["gp_ucb"]["mean"] == max(runs[b]["mean"] for b in BASELINES))
    # mutation must move the RCGP curve towards GP-UCB and away from the intact RCGP
    res["mutation_effect"] = {
        m: {"intact_mean": runs[f"{m}[Tc=zero,L=1.96]"]["mean"],
            "mutated_mean": runs[f"MUT_{m}[L=inf]"]["mean"],
            "gp_ucb_mean": runs["gp_ucb"]["mean"],
            "mutated_equals_gp_ucb": abs(runs[f"MUT_{m}[L=inf]"]["mean"] - runs["gp_ucb"]["mean"]) < 1e-9,
            "mutation_worsens_regret": runs[f"MUT_{m}[L=inf]"]["mean"] > runs[f"{m}[Tc=zero,L=1.96]"]["mean"]}
        for m in ("fc_rcgp_ucb", "a2_rcgp_ucb")}
    out[name] = res

# uncorrupted control (Figure 3)
unc = {}
for m in BASELINES:
    unc[m] = agg([run_bo(m, s, 30, corrupt=False)["cum_regret"] for s in SEEDS])
for m in ("fc_rcgp_ucb", "a2_rcgp_ucb"):
    for mode in ("est", "zero"):
        for lname, lval in (("q95", None), ("1.96", 1.96)):
            unc[f"{m}[Tc={mode},L={lname}]"] = agg(
                [run_bo(m, s, 30, corrupt=False, tc_mode=mode, fixed_L=lval)["cum_regret"]
                 for s in SEEDS])
out["uncorrupted_T30"] = {"cum_regret": unc,
                          "ranking_by_mean": [(k, round(unc[k]["mean"], 3))
                                              for k in sorted(unc, key=lambda k: unc[k]["mean"])]}

json.dump(out, open("results/claim5.json", "w"), indent=2)
brief = {k: ({kk: vv for kk, vv in v.items() if kk != "cum_regret"} if isinstance(v, dict) else v)
         for k, v in out.items()}
print(json.dumps(brief, indent=2))
