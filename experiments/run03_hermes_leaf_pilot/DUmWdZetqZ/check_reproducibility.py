"""check_reproducibility.py — the reproducibility gate for logbook.md.

Re-asserts EVERY number quoted in logbook.md against results/claim*.json.
Run:  .venv/bin/python check_reproducibility.py
Exit code 0 = every quoted number matches; 1 = at least one mismatch.

It does not re-run the experiments (per the no-re-derivation rule); it checks that the logbook
and the stored raw results agree. To regenerate the results:
    .venv/bin/python verify_claim1.py   (etc. for 2..5)
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda n: json.load(open(os.path.join(HERE, "results", n)))

fails, checks = [], 0


def eq(label, got, want, rel=1e-3, abs_=0.0):
    global checks
    checks += 1
    ok = (abs(got - want) <= max(abs_, rel * abs(want))) if isinstance(got, (int, float)) else got == want
    if not ok:
        fails.append(f"{label}: logbook says {want!r}, results give {got!r}")


def is_true(label, got):
    global checks
    checks += 1
    if not got:
        fails.append(f"{label}: expected True, got {got!r}")


# ------------------------------------------------------------------------ claim 1
c1 = R("claim1.json")
eq("c1 verdict", c1["verdict"], "verified")
eq("c1 grid points = 432", c1["T1_grid_points"], 432)
eq("c1 total violations = 12", c1["T1_violations"], 12)
is_true("c1 all violations have C > A", c1["T1_violations_all_have_C_gt_A"])
eq("c1 max B/Bmin among violations = 1.0", c1["T1_violation_B_over_Bmin_max"], 1.0)
eq("c1 subset B>=2Bmin: n = 360", c1["T1_subset_B_ge_2Bmin"]["n"], 360)
eq("c1 subset B>=2Bmin: 0 violations", c1["T1_subset_B_ge_2Bmin"]["violations"], 0)
eq("c1 subset B>=2Bmin: min slack 2.17", c1["T1_subset_B_ge_2Bmin"]["min_slack_ratio"], 2.17, rel=5e-3)
w = c1["T1_worst_point"]
eq("c1 worst point A", w["A"], 5.0); eq("c1 worst point C", w["C"], 200.0)
eq("c1 worst point Q", w["Q"], 1.0); eq("c1 worst point delta0", w["delta0"], 0.5)
eq("c1 worst point B", w["B"], 2460)
eq("c1 worst point exact err 9.730e-02", w["exact_err"], 9.730e-02, rel=1e-3)
eq("c1 worst point bound 7.134e-05", w["thm32_bound"], 7.134e-05, rel=1e-3)
eq("c1 worst slack ratio 7.33e-04", w["slack_ratio"], 7.33e-04, rel=5e-3)
eq("c1 T2 A", c1["T2_A"], 50.0); eq("c1 T2 C", c1["T2_C"], 20.0); eq("c1 T2 Q", c1["T2_Q"], 1.0)
eq("c1 T2 first budget 704", c1["T2_budgets"][0], 704)
eq("c1 T2 last budget 90078", c1["T2_budgets"][-1], 90078)
eq("c1 T2 first error 7.758e-01", c1["T2_exact_errors"][0], 7.758e-01, rel=1e-3)
eq("c1 T2 last error 1.604e-28", c1["T2_exact_errors"][-1], 1.604e-28, rel=1e-3)
eq("c1 T2 decay factor 4.84e27", c1["T2_error_ratio_first_to_last"], 4.84e27, rel=5e-3)
eq("c1 T2 log slope -6.820e-04", c1["T2_log_slope_per_sample"], -6.820e-04, rel=1e-3)
eq("c1 T2 R2 = 0.983", c1["T2_linear_fit_R2"], 0.983, rel=5e-3)
eq("c1 T3 exact 2.2946e-02", c1["T3_exact"], 2.2946e-02, rel=1e-3)
eq("c1 T3 mc mean 2.2659e-02", c1["T3_mc_mean"], 2.2659e-02, rel=1e-3)
eq("c1 T3 abs dev 2.87e-04", c1["T3_abs_dev_exact_vs_mc"], 2.87e-04, rel=5e-3)
eq("c1 M1 stalls at 0.582", c1["M1_constant_schedule"]["stalls_at"], 0.582, rel=5e-3)
eq("c1 M1 ratio first/last 1.0001", c1["M1_constant_schedule"]["error_ratio_first_to_last"], 1.0001, rel=1e-3)
eq("c1 M1 bound violations = 5", c1["M1_constant_schedule"]["n_bound_violations"], 5)
eq("c1 M2 stalls at 0.419", c1["M2_reversed_schedule"]["stalls_at"], 0.419, rel=5e-3)
eq("c1 M2 bound violations = 5", c1["M2_reversed_schedule"]["n_bound_violations"], 5)

# ------------------------------------------------------------------------ claim 2
c2 = R("claim2.json")
eq("c2 verdict", c2["verdict"], "verified")
eq("c2 grid points = 420", c2["n_grid_points"], 420)
eq("c2 max B_req/F = 24.41", c2["max_ratio_B_over_F_claimed"], 24.41, rel=1e-3)
byA = c2["max_ratio_claimed_by_A"]
for A, want in [("1.0", 24.41), ("10.0", 24.40), ("100.0", 24.14), ("1000.0", 20.94),
                ("10000.0", 15.45), ("100000.0", 13.66), ("1000000.0", 12.63)]:
    eq(f"c2 per-A max ratio A={A}", byA[A], want, rel=2e-3)
eq("c2 growth A1e6/A10 = 0.518", c2["ratio_claimed_growth_A1e6_over_A10"], 0.518, rel=5e-3)
byAn = c2["MUT_max_ratio_naive_by_A"]
for A, want in [("1.0", 24.41), ("10.0", 56.93), ("100.0", 83.38), ("1000.0", 105.62),
                ("10000.0", 126.89), ("100000.0", 147.83), ("1000000.0", 168.58)]:
    eq(f"c2 MUT per-A ratio A={A}", byAn[A], want, rel=2e-3)
eq("c2 MUT growth A1e6/A1 = 6.91", c2["MUT_ratio_naive_growth_A1e6_over_A1"], 6.91, rel=5e-3)
eq("c2 diag max B/T* (Q<=A) = 168.6", c2["max_ratio_B_over_Tstar_Q_le_A"], 168.6, rel=5e-3)
eq("c2 diag max B/(T*(1+lnT*)^2) = 13.6", c2["max_ratio_Tstar_over_ln2Tstar_Q_le_A"], 13.6, rel=5e-3)

# ------------------------------------------------------------------------ claim 3
c3 = R("claim3.json")
eq("c3 verdict", c3["verdict"], "verified")
eq("c3 delta0 = 1/(8e)", c3["delta0"], 1.0 / (8.0 * math.e))
eq("c3 f(delta0) = 100", c3["f_delta0"], 100)
eq("c3 grid size = 8", len(c3["grid"]), 8)
eq("c3 prop42 violations", c3["T1_prop42_violations"], 0)
eq("c3 prop42 vs delta violations", c3["T1_prop42_vs_delta_violations"], 0)
eq("c3 prop43 violations", c3["T2_prop43_violations"], 0)
eq("c3 prop43 vs delta violations", c3["T2_prop43_vs_delta_violations"], 0)
row01 = [r for r in c3["grid"] if abs(r["delta"] - 0.1) < 1e-12][0]
eq("c3 delta=0.1 L = 14", row01["L"], 14)
eq("c3 delta=0.1 exact err 2.321e-06", row01["exact_err"], 2.321e-06, rel=2e-3)
eq("c3 delta=0.1 prop42 bound 8.839e-02", row01["prop42_bound"], 8.839e-02, rel=1e-3)
eq("c3 delta=0.1 exact tail 1.120e-06", row01["exact_tail"], 1.120e-06, rel=2e-3)
eq("c3 delta=0.1 prop43 bound 6.104e-05", row01["prop43_bound"], 6.104e-05, rel=1e-3)
eq("c3 T3 slope 578.07", c3["T3_slope_A_empirical"], 578.07, rel=1e-3)
eq("c3 T3 predicted slope 577.08", c3["T3_slope_A_predicted"], 577.08, rel=1e-3)
eq("c3 T3 R2 = 0.99993", c3["T3_R2"], 0.99993, rel=1e-4)
eq("c3 MC L = 54", c3["MC_L"], 54)
eq("c3 MC err dev <= 2.4e-20", c3["MC_err_abs_dev"], 2.4e-20, abs_=1e-19)
eq("c3 MC tail dev <= 4.5e-22", c3["MC_tail_abs_dev"], 4.5e-22, abs_=1e-19)
eq("c3 M1 prop42 violations = 6", c3["M1_violations_of_prop42"], 6)
eq("c3 M1 error at largest L = 0.9935", c3["M1_error_at_largest_L"], 0.9935, rel=1e-3)
eq("c3 M1 ratio first/last = 0.486", c3["M1_error_ratio_first_to_last"], 0.486, rel=5e-3)

# ------------------------------------------------------------------------ claim 4
c4 = R("claim4.json")
eq("c4 verdict", c4["verdict"], "verified")
eq("c4 phase lengths Q=1", c4["T1_phase_lengths_depend_only_on_Q_and_i"]["Q=1"],
   [2, 4, 8, 16, 32, 64, 128])
is_true("c4 recommendation defined for every t", c4["T1_recommendation_defined_for_every_t"])
eq("c4 horizons checked = 80004", c4["T2_total_T_checked"], 80004)
eq("c4 propD3 violations", c4["T2_propD3_violations"], 0)
eq("c4 propD4 violations", c4["T2_propD4_violations"], 0)
eq("c4 thmD5 violations", c4["T3_violations"], 0)
eq("c4 thmD5 settings x horizons = 21", sum(len(s["rows"]) for s in c4["T3_thmD5"]), 21)
is_true("c4 min error decay >= 4.29e9", c4["T3_min_err_ratio_first_to_last"] >= 4.29e9)
eq("c4 M1 bound violations = 4", c4["M1_constant_phase_length"]["n_violations"], 4)
eq("c4 M1 error stalls at 1.0", c4["M1_constant_phase_length"]["err_stalls_at"], 1.0)
eq("c4 M2 propD4 violations = 6", c4["M2_slow_growth_1p05"]["n_propD4_violations"], 6)
eq("c4 M2 horizons checked = 6", len(c4["M2_slow_growth_1p05"]["rows"]), 6)

# ------------------------------------------------------------------------ claim 5
c5 = R("claim5.json")
eq("c5 A verdict", c5["A_verdict"], "verified")
eq("c5 mu", c5["instance"]["mu"], [1.0, 0.6, 0.4, 0.2])
eq("c5 sigma", c5["instance"]["sigma"], [0.15, 0.15, 1.0, 1.0])
eq("c5 A_thm51 = 295.78", c5["instance"]["A_thm51"], 295.78, rel=1e-4)
eq("c5 C_thm51 = 956.13", c5["instance"]["C_thm51"], 956.13, rel=1e-4)
eq("c5 A1 stopping violations", c5["A1_violations_stopping"], 0)
eq("c5 A1 correctness violations", c5["A1_violations_correctness"], 0)
eq("c5 A1 deltas checked = 3", len(c5["A1_thm51"]), 3)
a3 = c5["A3_fc2fb_pekhn"]
eq("c5 A3 budgets", [r["B"] for r in a3], [2500, 5000, 10000, 20000, 40000])
for i, want in enumerate([0.724, 0.050, 0.0070, 0.0, 0.0]):
    eq(f"c5 A3 mean err B={a3[i]['B']}", a3[i]["mean_err"], want, rel=1e-2, abs_=1e-6)
for i, want in enumerate([2.488, 2.127, 1.588, 0.919, 0.329]):
    eq(f"c5 A3 thm32 bound B={a3[i]['B']}", a3[i]["thm32_bound"], want, rel=2e-3)
eq("c5 A3 violations", c5["A3_violations_thm32"], 0)
is_true("c5 A3 monotone decreasing", c5["A3_err_decreasing"])
is_true("c5 A3 shows decay", c5["A3_shows_decay"])
eq("c5 A4 ours expr", c5["A4_symbolic"]["ours_expr"], "K**7 - 2*K**6 + 2*K**5")
eq("c5 A4 shvar expr", c5["A4_symbolic"]["shvar_expr"], "K**9 - 2*K**8 + 2*K**5")
eq("c5 A4 ours degree = 7", c5["A4_symbolic"]["ours_leading_degree"], 7)
eq("c5 A4 shvar degree = 9", c5["A4_symbolic"]["shvar_leading_degree"], 9)
is_true("c5 A4 matches paper", c5["A4_matches_paper"])
m1 = c5["M1_noise_blind"]
eq("c5 M1 ratio delta=0.1 = 4.85", m1[0]["ratio"], 4.85, rel=2e-3)
eq("c5 M1 ratio delta=0.01 = 4.78", m1[1]["ratio"], 4.78, rel=2e-3)
is_true("c5 M1 blind is worse", c5["M1_blind_is_worse"])
eq("c5 A2 points where cor52 stronger = 9", c5["A2_n_points_where_cor52_stronger_than_thm32"], 9)
eq("c5 A2 max ratio thm32/cor52 = 3.29e28", c5["A2_max_ratio_thm32_over_cor52"], 3.29e28, rel=5e-3)
eq("c5 log2/ln factor 1.4427", 1.0 / math.log(2.0), 1.4427, rel=1e-4)
eq("c5 B linear verdict", c5["B_linear_bandits_cor54"]["verdict"], "inconclusive")
eq("c5 C unimodal verdict", c5["C_unimodal_cor57"]["verdict"], "inconclusive")

# ------------------------------------------------------------------------------ report
print(f"checks run: {checks}")
if fails:
    print(f"FAILED: {len(fails)}")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL LOGBOOK NUMBERS REPRODUCE AGAINST results/*.json")
sys.exit(0)
