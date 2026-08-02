"""check_reproducibility.py -- reproducibility gate for logbook.md.

Re-asserts every number quoted in logbook.md against results/*.json.
Exit code 0 = all quoted numbers match the stored raw outputs; 1 = drift.

Usage:  python check_reproducibility.py
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda name: json.load(open(os.path.join(HERE, "results", name)))
fails, checks = [], 0


def eq(label, got, want, tol=0.0):
    global checks
    checks += 1
    ok = (got == want) if tol == 0 else (abs(got - want) <= tol)
    if not ok:
        fails.append(f"{label}: logbook says {want}, results say {got}")


def le(label, got, want):
    global checks
    checks += 1
    if not got <= want:
        fails.append(f"{label}: {got} !<= {want}")


# ----------------------------- claim 1 -------------------------------------
c1 = R("claim1.json")
eq("c1.alpha", c1["alpha"], 0.1)
eq("c1.K", c1["K"], 3)
eq("c1.n_cal", c1["n_cal_per_source"], 50)
eq("c1.trials", c1["n_trials_per_world"], 4000)
eq("c1.worlds", c1["main"]["n_worlds"], 8)
eq("c1.worst_source_coverage_min", round(c1["main"]["worst_source_coverage_min"], 4), 0.9767)
eq("c1.worst_source_coverage_mean", round(c1["main"]["worst_source_coverage_mean"], 4), 0.9861)
eq("c1.worlds_below_target", c1["main"]["n_worlds_below_target"], 0)
eq("c1.union_identity_mismatches", c1["union_identity_total_mismatches"], 0)
eq("c1.mixture_grid_size", c1["mixture"]["n_mixture_weights"], 968)
eq("c1.worst_mixture_coverage", round(c1["mixture"]["worst_mixture_coverage"], 4), 0.9767)
eq("c1.mut_no_plus_one", round(c1["mutation_no_plus_one"]["worst_source_coverage_min"], 4), 0.9662)
eq("c1.mut_mean_p", round(c1["mutation_mean_p"]["worst_source_coverage_min"], 4), 0.9520)
eq("c1.mut_single_source", round(c1["mutation_single_source"]["worst_source_coverage_min"], 4), 0.5514)
eq("c1.mut_single_source_mean", round(c1["mutation_single_source"]["worst_source_coverage_mean"], 4), 0.6460)
eq("c1.mut_single_worlds_below", c1["mutation_single_source"]["n_worlds_below_target"], 8)
eq("c1.mean_set_size", round(c1["main"]["mean_set_size"], 2), 19.20)

# ----------------------------- claim 2 -------------------------------------
c2 = R("claim2.json")
eq("c2.n_instances", c2["n_instances"], 2880)
le("c2.max_size_gap", c2["max_size_gap"], 1e-12)
le("c2.max_strong_duality_gap", c2["max_strong_duality_gap"], 1e-12)
eq("c2.superlevel_matches_lp", c2["n_superlevel_matches_lp"], 2880)
eq("c2.superlevel_feasible", c2["n_superlevel_feasible"], 2880)
eq("c2.cs_i", c2["n_cs_i"], 2880)
eq("c2.cs_ii", c2["n_cs_ii"], 2880)
eq("c2.cs_iii", c2["n_cs_iii"], 2880)
eq("c2.at_least_one_exact", c2["n_at_least_one_exact_coverage"], 2880)
eq("c2.mut_uniform_lambda_bad", c2["mut_uniform_lambda_infeasible_or_larger"], 2880)
eq("c2.mut_uniform_lambda_infeasible", c2["mut_uniform_lambda_infeasible"], 2680)
eq("c2.mut_inverted_infeasible", c2["mut_inverted_superlevel_infeasible"], 2880)
eq("c2.mut_drop_active", c2["mut_drop_active_constraint_strictly_smaller"]
   if "mut_drop_active_constraint_strictly_smaller" in c2 else c2["mut_drop_active_strictly_smaller"], 2847)

# ----------------------------- claim 3 -------------------------------------
c3 = R("claim3.json")
eq("c3.rho_T_mean", round(c3["rho_T_mean"], 3), 2.875)
eq("c3.worlds", c3["worlds"], 8)
eq("c3.sample_sizes", c3["sample_sizes"], [100, 400, 1600, 6400, 25600])
eq("c3.symdiff_n100", round(c3["main"]["100"]["mean_symdiff_outside_T"], 4), 1.2823)
eq("c3.symdiff_n25600", round(c3["main"]["25600"]["mean_symdiff_outside_T"], 4), 0.0546)
eq("c3.mut_uniform_lambda_n25600", round(c3["mutation_frozen_uniform_lambda"]["25600"]["mean_symdiff_outside_T"], 4), 0.3577)
eq("c3.mut_single_density_n25600", round(c3["mutation_single_source_density_score"]["25600"]["mean_symdiff_outside_T"], 4), 1.6233)
for n in ["100", "400", "1600", "6400", "25600"]:
    eq(f"c3.thm3_bound_ok_n{n}", c3["main"][n]["n_worlds_satisfying_thm3_bound"], 8)
    checks += 1
    if c3["main"][n]["min_worst_source_coverage"] < 0.87:
        fails.append(f"c3.min_worst_source_coverage n={n}: {c3['main'][n]['min_worst_source_coverage']} < 0.87")

# --------------------------- claims 4/5/6 ----------------------------------
c45 = R("claim4_5.json")
eq("c45.verdict", c45["verdict"], "inconclusive")
eq("c45.trials_completed", c45["attempt"]["trials_completed"], 0)
eq("c45.measured_reduction_is_null", c45["measured_percent_reduction"], None)
eq("c45.wall_time_before_stop_s", c45["attempt"]["wall_time_seconds_before_stop"], 1780)
c6 = R("claim6.json")
eq("c6.verdict", c6["verdict"], "inconclusive")
eq("c6.n_countries", c6["paper_side_facts_checked_in_text"]["n_countries_in_fmow_training_text"], 249)
eq("c6.n_regions", c6["paper_side_facts_checked_in_text"]["n_regions"], 6)

print(f"checked {checks} quoted values")
if fails:
    print("FAILED:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL LOGBOOK NUMBERS REPRODUCE FROM results/*.json")
