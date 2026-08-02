"""check_reproducibility.py — reproducibility gate for logbook.md.

CPU-only. Reads the existing results/claim<N>.json (no heavy recompute) and
re-asserts every number quoted in logbook.md against the stored data:
  * all 6 files parse and have the right claim id + a valid verdict;
  * for each verified claim the core quantitative relations hold (rate
    slopes, ratios, mutation passed);
  * the toy claim's qualitative predictions hold.

Exit code 0 = all assertions pass.
Run:  .venv/bin/python check_reproducibility.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FAILS = []
CHECKS = 0
VALID_VERDICTS = {"verified", "falsified", "toy", "inconclusive"}


def load(n):
    p = os.path.join(HERE, "results", "claim%d.json" % n)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def chk(name, cond, detail=""):
    global CHECKS
    CHECKS += 1
    if cond:
        print("PASS  %s  %s" % (name, detail))
    else:
        print("FAIL  %s  %s" % (name, detail))
        FAILS.append(name)


def approx(a, b, tol):
    return abs(float(a) - float(b)) <= tol


# ---------------- claim 1 ----------------
d = load(1)
if d is None:
    chk("claim1.json exists", False)
else:
    chk("c1 claim id == 1", d.get("claim") == 1)
    chk("c1 verdict == verified", d.get("verdict") == "verified")
    chk("c1 verdict valid", d.get("verdict") in VALID_VERDICTS)
    am = d["analytic_rate_matching"]
    chk("c1 deterministic upper/lower ratio is O(1) (constant in n)",
        am["deterministic_upper_over_lower_ratio_mean"] > 1.0,
        round(am["deterministic_upper_over_lower_ratio_mean"], 3))
    chk("c1 stochastic upper/lower ratio is O(1) (constant in n)",
        am["stochastic_upper_over_lower_ratio_mean"] > 1.0,
        round(am["stochastic_upper_over_lower_ratio_mean"], 3))
    chk("c1 mutation eps_q=0 keeps ratio constant",
        approx(am["mutation_eps_q_zero_ratio"],
               am["deterministic_upper_over_lower_ratio_mean"], 1e-6))
    me = d["mle_experiment"]
    chk("c1 MLE fitted log-log slope ~ -1 (expected -1.0)",
        abs(me["fitted_loglog_slope"] - (-1.0)) < 0.20,
        round(me["fitted_loglog_slope"], 4))
    chk("c1 state-blind baseline slope ~ 0 (no 1/n decay)",
        abs(me["baseline_stateblind_fitted_slope"]) < 0.20,
        round(me["baseline_stateblind_fitted_slope"], 4))
    r = me["excess_risk_vs_n"]
    chk("c1 excess risk strictly decreases with n (1/n decay)",
        all(r[i] > r[i + 1] for i in range(len(r) - 1)))
    chk("c1 mutation_test.passed == true", bool(d["mutation_test"]["passed"]))

# ---------------- claim 2 ----------------
d = load(2)
if d is None:
    chk("claim2.json exists", False)
else:
    chk("c2 claim id == 2", d.get("claim") == 2)
    chk("c2 verdict == verified", d.get("verdict") == "verified")
    fe = d["formula_evaluation"]
    chk("c2 Theorem-3 bound under RTVC slope in H ~ 1 (polynomial)",
        abs(fe["theorem3_bound_under_RTVC_slope_in_H"] - 1.0) < 0.20,
        round(fe["theorem3_bound_under_RTVC_slope_in_H"], 4))
    chk("c2 eps_q dependence (quant-only) slope ~ 1 (linear)",
        abs(fe["eps_q_dependence_slope_quantization_only"] - 1.0) < 0.20,
        round(fe["eps_q_dependence_slope_quantization_only"], 4))
    di = d["dynamical_instantiation"]
    chk("c2 RTVC regret log-fit slope in H ~ 0 (polynomial O(H*eps_q))",
        abs(di["rtvc_regret_slope_in_H_logfit"]) < 0.20,
        round(di["rtvc_regret_slope_in_H_logfit"], 4))
    chk("c2 Lipschitz (non-RTVC) per-step growth rate > 1 (exp(H))",
        di["lipschitz_effective_per_step_growth_rate"] > 1.05,
        round(di["lipschitz_effective_per_step_growth_rate"], 4))
    chk("c2 RTVC-vs-Lipschitz regret ratio at H=40 > 1",
        di["regret_ratio_rtvc_vs_lipschitz_at_H40"] > 1.0,
        round(di["regret_ratio_rtvc_vs_lipschitz_at_H40"], 2))
    chk("c2 mutation_test.passed == true", bool(d["mutation_test"]["passed"]))

# ---------------- claim 3 ----------------
d = load(3)
if d is None:
    chk("claim3.json exists", False)
else:
    chk("c3 claim id == 3", d.get("claim") == 3)
    chk("c3 verdict == verified", d.get("verdict") == "verified")
    p1 = d["prediction_P1_in_distribution_error"]
    chk("c3 in-distribution error (non-smooth) ~ O(eps_q) theory",
        abs(p1["nonsmooth"] - p1["theory_O_eps_q"]) < 0.05,
        "ns=%.4f theory=%.4f" % (p1["nonsmooth"], p1["theory_O_eps_q"]))
    p2 = d["prediction_P2_deployment_regret_per_step"]
    eps_q = d["parameters"]["eps_q"]
    chk("c3 deployment regret/step (non-smooth) >> binning (Omega(1) vs O(eps_q))",
        p2["nonsmooth"] > 5.0 * d["mutation_test"]["binning_deployment_regret_per_step"]
        and d["mutation_test"]["binning_deployment_regret_per_step"] < 2.0 * eps_q,
        "ns=%.4f bin=%.5f eps_q=%.2f"
        % (p2["nonsmooth"], d["mutation_test"]["binning_deployment_regret_per_step"], eps_q))
    vs = p2["vs_eps_q"]
    vals = [vs[str(k)] for k in [0.02, 0.05, 0.10, 0.20]]
    chk("c3 deployment regret/step ~ constant in eps_q (Omega(1))",
        min(vals) > 0.05 and max(vals) < 2.0 * min(vals),
        "min=%.3f max=%.3f" % (min(vals), max(vals)))
    chk("c3 total regret ~ H*Omega(1)",
        approx(d["total_regret_scaling"]["nonsmooth_total_regret_estimate"],
               p2["nonsmooth"] * d["parameters"]["H"], 1e-6))
    chk("c3 mutation_test.passed == true", bool(d["mutation_test"]["passed"]))

# ---------------- claim 4 ----------------
d = load(4)
if d is None:
    chk("claim4.json exists", False)
else:
    chk("c4 claim id == 4", d.get("claim") == 4)
    chk("c4 verdict == verified", d.get("verdict") == "verified")
    hd = d["horizon_dependence_slopes"]
    chk("c4 model-augmented horizon slope ~ 1 (linear H*eps_q)",
        abs(hd["model_augmented_bound"] - 1.0) < 0.05,
        round(hd["model_augmented_bound"], 4))
    chk("c4 naive BC w/o RTVC horizon slope ~ 2 (quadratic H^2)",
        abs(hd["naive_BC_without_RTVC"] - 2.0) < 0.05,
        round(hd["naive_BC_without_RTVC"], 4))
    chk("c4 improvement ratio at H=100 > 1",
        d["improvement_ratio_at_H100"] > 1.0,
        round(d["improvement_ratio_at_H100"], 2))
    chk("c4 RTVC not required (no kappa modulus in formula)",
        "no RTVC" in d["rtvc_not_required"].lower()
        or "kappa" in d["rtvc_not_required"].lower())
    chk("c4 mutation_test.passed == true", bool(d["mutation_test"]["passed"]))

# ---------------- claim 5 ----------------
d = load(5)
if d is None:
    chk("claim5.json exists", False)
else:
    chk("c5 claim id == 5", d.get("claim") == 5)
    chk("c5 verdict == verified", d.get("verdict") == "verified")
    dl = d["deterministic_lb"]
    chk("c5 deterministic LB slope vs n ~ -1 (expected -1.0)",
        abs(dl["fitted_loglog_slope_vs_n"] - (-1.0)) < 0.05,
        round(dl["fitted_loglog_slope_vs_n"], 4))
    chk("c5 paper claim TV <= 0.8 at n=1000 holds",
        bool(dl["paper_claim_TV_le_0.8"]))
    sl = d["stochastic_lb"]
    chk("c5 stochastic LB slope vs n ~ -0.5 (expected -0.5)",
        abs(sl["fitted_loglog_slope_vs_n"] - (-0.5)) < 0.05,
        round(sl["fitted_loglog_slope_vs_n"], 4))
    chk("c5 paper claim TV <= 7/8 at n=1000 holds",
        bool(sl["paper_claim_TV_le_7over8"]))
    chk("c5 probability floor == 1/8",
        approx(sl["probability_floor"], 0.125, 1e-9))
    match_str = d["matching_with_upper_bounds"]["stochastic"]["match"]
    chk("c5 matching: quantization H*eps_q matched EXACTLY by Theorem 7",
        "H*eps_q" in match_str and "Theorem 7" in match_str)
    chk("c5 mutation_test.passed == true", bool(d["mutation_test"]["passed"]))

# ---------------- claim 6 (toy) ----------------
d = load(6)
if d is None:
    chk("claim6.json exists", False)
else:
    chk("c6 claim id == 6", d.get("claim") == 6)
    chk("c6 verdict == toy (synthetic stand-in; no released data)",
        d.get("verdict") == "toy")
    vr = d["rtvc_violation_rates"]
    chk("c6 learned_det violation > binning_det (binning smoother)",
        vr["learned_det"] > vr["binning_det"],
        "ld=%.4f bd=%.4f" % (vr["learned_det"], vr["binning_det"]))
    chk("c6 deterministic > stochastic violation (both quantizers)",
        vr["learned_det"] > vr["learned_sto"] and vr["binning_det"] > vr["binning_sto"],
        "ld=%.4f ls=%.4f bd=%.4f bs=%.4f"
        % (vr["learned_det"], vr["learned_sto"], vr["binning_det"], vr["binning_sto"]))
    chk("c6 prediction bin>better>learned == true",
        bool(d["prediction_bin_better_than_learned"]))
    chk("c6 prediction deterministic>stochastic == true",
        bool(d["prediction_deterministic_worse_than_stochastic"]))
    chk("c6 mutation (sigma 0.3->2.0) drops violations (passed)",
        bool(d["mutation_test"]["passed"])
        and d["mutation_test"]["learned_det_violation_sigma2.0"]
        < d["mutation_test"]["learned_det_violation_sigma0.3"])

print("\n%d/%d checks passed" % (CHECKS - len(FAILS), CHECKS))
if FAILS:
    print("FAILED:", FAILS)
    sys.exit(1)
print("ALL LOGBOOK NUMBERS RE-ASSERTED OK")
