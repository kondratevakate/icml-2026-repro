"""check_reproducibility.py — reproducibility gate for logbook.md.

Re-asserts every number quoted in logbook.md against results/*.json.
Run: .venv/bin/python check_reproducibility.py     (exit 0 = all assertions pass)
"""
import json
import os
import sys

R = "results"
fails, checks = [], 0


def load(n):
    with open(os.path.join(R, f"claim{n}.json")) as f:
        return json.load(f)


def ck(desc, cond, got=None):
    global checks
    checks += 1
    if not cond:
        fails.append(f"FAIL: {desc} (got {got})")
    else:
        print(f"ok  : {desc}" + (f"  [{got}]" if got is not None else ""))


# ---------------- claim 1 ----------------
c1 = load(1)
b = c1["b_nonlinearity"]
ck("c1: min max chord gap over 45 BoN instances >= 1e-4",
   b["min_max_chord_gap_over_instances"] >= 1e-4, b["min_max_chord_gap_over_instances"])
ck("c1: max chord gap <= 0.082", b["max_max_chord_gap_over_instances"] <= 0.082,
   b["max_max_chord_gap_over_instances"])
ck("c1: linear reward chord gap <= 1e-15", b["linear_reward_max_chord_gap"] <= 1e-15,
   b["linear_reward_max_chord_gap"])
ck("c1 MUTATION N=1: chord gap <= 1e-15", b["MUTATION_N1_max_chord_gap"] <= 1e-15,
   b["MUTATION_N1_max_chord_gap"])
a = c1["c_adaptation"]
ck("c1: 60/60 instances IAMA base beats naive base on the IAMA objective",
   a["n_instances"] == 60 and a["n_instances_iama_better"] == 60, a["n_instances_iama_better"])
ck("c1: min improvement >= 5.7e-3", a["min_improvement"] >= 5.7e-3, a["min_improvement"])
ck("c1 MUTATION N=1: TV(IAMA opt, naive opt) <= 1e-9",
   c1["MUTATION_N1_tv_between_iama_and_naive_optimum"]["max"] <= 1e-9,
   c1["MUTATION_N1_tv_between_iama_and_naive_optimum"]["max"])
ck("c1: mirror-descent vs L-BFGS TV <= 1e-7", c1["sanity_tv_mirror_descent_vs_lbfgs"] <= 1e-7,
   c1["sanity_tv_mirror_descent_vs_lbfgs"])

# ---------------- claim 2 ----------------
c2 = load(2)
t1 = c2["T1_functional_derivative_vs_finite_difference"]
ck("c2: 80 finite-difference cases", t1["n_cases"] == 80, t1["n_cases"])
ck("c2: max abs FD error <= 1e-8", t1["max_abs_err"] <= 1e-8, t1["max_abs_err"])
ck("c2: max rel FD error <= 1e-6", t1["max_rel_err"] <= 1e-6, t1["max_rel_err"])
t2 = c2["T2_dropin_and_mutation"]
ck("c2: 30 drop-in instances", t2["n_instances"] == 30, t2["n_instances"])
ck("c2: non-linear GRPO suboptimality <= 1e-12", t2["max_suboptimality_nonlinear_grpo"] <= 1e-12,
   t2["max_suboptimality_nonlinear_grpo"])
ck("c2: TV to independent optimum <= 1e-7", t2["max_tv_to_optimum"] <= 1e-7, t2["max_tv_to_optimum"])
ck("c2 MUTATION plain reward: worse in 30/30 instances",
   t2["n_instances_mutation_strictly_worse"] == 30, t2["n_instances_mutation_strictly_worse"])
ck("c2 MUTATION: min excess loss >= 5.4e-3", t2["min_MUTATION_excess_loss"] >= 5.4e-3,
   t2["min_MUTATION_excess_loss"])
t3 = c2["T3_empirical_derivative_error"]
ck("c2: log-log slope of E||dhat-d||_sp^2 vs M in [-1.15,-0.85]",
   -1.15 <= t3["loglog_slope_vs_M"] <= -0.85, t3["loglog_slope_vs_M"])

# ---------------- claim 3 ----------------
c3 = load(3)
ck("c3: 90 instances x 25 iterates", c3["n_instances"] == 90 and c3["n_rows_total"] == 2250,
   (c3["n_instances"], c3["n_rows_total"]))
ck("c3: concavity (Assumption 5.1) violation == 0", c3["max_concavity_violation"] == 0.0,
   c3["max_concavity_violation"])
ck("c3: no bound violation with excess loss above 1e-12",
   c3["n_rows_bound_violated_above_1e-12"] == 0, c3["n_rows_bound_violated_above_1e-12"])
ck("c3: largest excess loss among raw violations <= 1e-15 (float64 noise)",
   c3["max_lhs_among_violating_rows"] <= 1e-15, c3["max_lhs_among_violating_rows"])
ck("c3: max final (T=25) excess loss <= 1e-5", c3["max_final_excess_loss"] <= 1e-5,
   c3["max_final_excess_loss"])
m1 = c3["MUTATION_eta_20_over_L"]
ck("c3 MUTATION eta=20/L: >=5 macroscopic bound violations (excess loss > 1e-3)",
   m1["n_violating_macroscopically"] >= 5, m1["n_violating_macroscopically"])
m2 = c3["MUTATION_plain_reward_standard_grpo"]
ck("c3 MUTATION plain reward: all 36 cases violate the bound, >=35 macroscopically",
   m2["n"] == 36 and m2["n_violating"] == 36 and m2["n_violating_macroscopically"] >= 35,
   (m2["n_violating"], m2["n_violating_macroscopically"]))
ck("c3 MUTATION plain reward: min final excess loss >= 9e-4",
   m2["min_final_excess_loss"] >= 9e-4, m2["min_final_excess_loss"])

# ---------------- claim 4 ----------------
c4 = load(4)
ck("c4: 60 configs", c4["n_configs"] == 60, c4["n_configs"])
ck("c4: Theorem 5.3 full bound never violated",
   c4["n_configs_full_bound_violated"] == 0, c4["n_configs_full_bound_violated"])
ck("c4: max LHS/RHS ratio < 1", c4["max_ratio_lhs_over_full_bound"] < 1.0,
   c4["max_ratio_lhs_over_full_bound"])
mm = c4["MUTATION_drop_bias_term"]
ck("c4 MUTATION (drop 2(eps+delta)/beta): decay-only bound violated in >=1 noisy config",
   mm["n_violating_decay_only_bound"] >= 1, mm["n_violating_decay_only_bound"])
bf = c4["bias_floor_monotone_in_noise"]
ck("c4: bias floor monotone increasing in noise in all 12 (K,N,beta) cells",
   bf["n_monotone"] == bf["n"], (bf["n_monotone"], bf["n"]))

# ---------------- claim 5 ----------------
c5 = load(5)
ck("c5: CDF derivative == closed-form density (sympy)",
   c5["T1_cdf_derivative_matches_density_symbolically"] is True)
for N, m in c5["T1_numeric_mass"].items():
    ck(f"c5: closed-form density integrates to 1 within 2e-3 (N={N})", abs(m - 1) < 2e-3, m)
for N, s in c5["T2_relative_span_of_aggregate_derivative"].items():
    ck(f"c5: aggregate functional derivative constant at pi* (N={N}), rel span <= 1e-14",
       s <= 1e-14, s)
for N, s in c5["M2_wrong_alpha_relative_span"].items():
    ck(f"c5 MUTATION alpha=1/N (N={N}): rel span >= 0.03", s >= 0.03, s)
ck("c5 MUTATION N=2: closed form is exactly uniform",
   c5["M1_N2_max_abs_density_minus_1"] <= 1e-12, c5["M1_N2_max_abs_density_minus_1"])
for N, d in c5["T3_grid_optimum_vs_closed_form"].items():
    ck(f"c5: mirror-ascent objective matches closed form within 2e-5 (N={N})",
       d["max_abs_objective_gap"] <= 2e-5, d["max_abs_objective_gap"])
ck("c5: N=2 grid optimum TV to closed form <= 1e-12 from the uniform init",
   c5["T3_grid_optimum_vs_closed_form"]["2"]["per_seed"][0]["tv_distance"] <= 1e-12,
   c5["T3_grid_optimum_vs_closed_form"]["2"]["per_seed"][0]["tv_distance"])
t4 = c5["T4_naive"]
ck("c5: naive optimum concentrates at y=0.5 (mass within 0.01 of 0.5 == 1.0)",
   abs(t4["mass_within_0.01_of_0.5"] - 1.0) < 1e-9, t4["mass_within_0.01_of_0.5"])
ck("c5: naive optimum std <= 2e-3", t4["std"] <= 2e-3, t4["std"])
ck("c5: IAMA (N=8) puts >= 0.92 mass outside [0.25,0.75]; naive puts 0.0",
   c5["T4_iama_N8_mass_outside_center"] >= 0.92 and c5["T4_naive_mass_outside_center"] < 1e-12,
   (c5["T4_iama_N8_mass_outside_center"], c5["T4_naive_mass_outside_center"]))

# ---------------- claim 6 ----------------
c6 = load(6)
ck("c6: verdict inconclusive, no toy substitute",
   c6["verdict"] == "inconclusive" and c6["toy_substitute_used"] is False)

print(f"\n{checks - len(fails)}/{checks} assertions passed")
if fails:
    print("\n".join(fails))
    sys.exit(1)
print("ALL LOGBOOK NUMBERS REPRODUCED")
