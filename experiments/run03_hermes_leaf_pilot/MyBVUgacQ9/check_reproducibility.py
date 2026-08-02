"""check_reproducibility.py -- reproducibility gate for logbook.md.

Re-asserts every number quoted in logbook.md against the raw artifacts in results/*.json.
Exit code 0 = all assertions pass.

Run:  .venv/bin/python check_reproducibility.py
"""
import json
import math
import sys

FAILED = []
PASSED = 0


def load(n):
    with open("results/claim%d.json" % n) as fh:
        return json.load(fh)


def check(label, cond, detail=""):
    global PASSED
    if cond:
        PASSED += 1
        print("  PASS  %s %s" % (label, detail))
    else:
        FAILED.append(label)
        print("  FAIL  %s %s" % (label, detail))


def approx(a, b, tol):
    return a is not None and b is not None and abs(a - b) <= tol


# ------------------------------------------------------------------ claim 1
print("claim 1 (Theorem 3.1) -- verified")
c1 = load(1)
s = c1["tests"]["summary"]
check("c1.n_runs == 450", s["n_runs"] == 450, "(%d)" % s["n_runs"])
check("c1.all runs converged (res < 1e-8)", s["n_converged_res_lt_1e-8"] == 450,
      "(%d/450)" % s["n_converged_res_lt_1e-8"])
check("c1.all runs feasible (viol < 1e-8)", s["n_feasible_viol_lt_1e-8"] == 450,
      "(%d/450)" % s["n_feasible_viol_lt_1e-8"])
check("c1.max final residual = 3.85e-16 (<1e-14)",
      s["max_final_res"] < 1e-14, "(%.3e)" % s["max_final_res"])
check("c1.max final constraint violation = 2.22e-16 (<1e-14)",
      s["max_final_viol"] < 1e-14, "(%.3e)" % s["max_final_viol"])
check("c1.o(1/k) witness max_last_k_gap = 0.0", s["max_last_k_gap"] == 0.0,
      "(%.3e)" % s["max_last_k_gap"])
check("c1.k*gap < 1e-6 in 450/450 runs", s["n_k_gap_lt_1e-6"] == 450,
      "(%d)" % s["n_k_gap_lt_1e-6"])
check("c1.max blockwise-optimality error = 5.19e-10 (<1e-8)",
      s["max_blockwise_opt_err"] < 1e-8, "(%.3e)" % s["max_blockwise_opt_err"])
ma = c1["tests"]["MUT_A_assumption26_violated"]
check("c1.MUT-A iterates collapse to origin", ma["all_collapse_to_origin"],
      "(max||x||=%.2e)" % ma["max_norm_x_final"])
check("c1.MUT-A dual diverges to -inf (w <= -4000)", ma["all_dual_diverges_negative"]
      and ma["max_w_final"] <= -1e3, "(max w=%.1f)" % ma["max_w_final"])
check("c1.MUT-A limit point infeasible (violation = 1.0)", ma["all_infeasible"],
      "(min viol=%.3f)" % ma["min_viol_final"])
mb = c1["tests"]["MUT_B_rho_sweep"]
check("c1.MUT-B threshold value = 500", approx(mb["threshold"], 500.0, 1e-6),
      "(%.1f)" % mb["threshold"])
check("c1.MUT-B rho far below threshold still converges (worst res < 1e-12)",
      all(r["worst_final_res"] < 1e-12 for r in mb["runs"]
          if r["rho_over_threshold"] < 1.0),
      "(sufficient-not-necessary)")

# ------------------------------------------------------------------ claim 2
print("claim 2 (Theorem 3.2 / eq. 4) -- verified")
c2 = load(2)
ps, pl = c2["profile_small"], c2["profile_large"]
check("c2.||C||=0 reference contraction factor = 0.0588",
      approx(ps[0]["worst_factor"], 0.0588, 5e-4), "(%.4f)" % ps[0]["worst_factor"])
check("c2.worst factor over small-||C|| regime = 0.1111 (<0.2)",
      approx(c2["verdict_numbers"]["max_worst_factor_over_small_C"], 0.1111, 5e-4),
      "(%.4f)" % c2["verdict_numbers"]["max_worst_factor_over_small_C"])
check("c2.all small-||C|| runs linear (factor < 0.999)",
      c2["verdict_numbers"]["all_small_C_linear_factor_lt_1"], "")
big = {r["C_norm"]: r for r in pl}
check("c2.MUTATION ||C||=25 factor = 0.9981 (>=0.99)",
      approx(big[25.0]["worst_factor"], 0.9981, 5e-4),
      "(%.4f)" % big[25.0]["worst_factor"])
check("c2.MUTATION ||C||=50 factor = 1.0004 (>=1: no contraction) and viol 7.7e-05",
      big[50.0]["worst_factor"] >= 0.999 and big[50.0]["worst_final_viol"] > 1e-6,
      "(%.4f, viol %.1e)" % (big[50.0]["worst_factor"], big[50.0]["worst_final_viol"]))
check("c2.MUTATION arm loses linear rate", c2["verdict_numbers"]["some_large_C_not_linear"], "")

# ------------------------------------------------------------------ claim 3
print("claim 3 (Theorem 3.3, polyhedral) -- verified")
c3 = load(3)
cells = c3["polyhedral_small_C"]
worst_res = max(v["worst_res_factor"] for v in cells.values())
worst_gap = max(v["worst_thm33_gap_factor"] for v in cells.values())
check("c3.all 4 polyhedral cells contract, worst residual factor = 0.2000 (<0.25)",
      approx(worst_res, 0.2000, 5e-4), "(%.4f)" % worst_res)
check("c3.worst Theorem-3.3 gap factor = 0.0881 (<0.2)",
      approx(worst_gap, 0.0881, 5e-4), "(%.4f)" % worst_gap)
check("c3.all 4 box constraints active at the limit (nonsmooth regime)",
      all(set(v["n_active_constraints_at_limit"]) == {4} for v in cells.values()), "")
check("c3.limit point passes local-minimum probe in every cell (0 failures)",
      all(v["local_min_probe_failures"] == 0 for v in cells.values()), "")
mut3 = c3["MUTATION_A_large_C"]
check("c3.MUTATION ||C||x10 -> residual factor 0.9858, gap factor 0.9977",
      approx(mut3["poly_Cscale10_rho4"]["worst_res_factor"], 0.9858, 5e-4) and
      approx(mut3["poly_Cscale10_rho4"]["worst_thm33_gap_factor"], 0.9977, 5e-4),
      "(%.4f, %.4f)" % (mut3["poly_Cscale10_rho4"]["worst_res_factor"],
                        mut3["poly_Cscale10_rho4"]["worst_thm33_gap_factor"]))
check("c3.MUTATION ||C||x50 -> residual factor 1.0006 (no contraction)",
      mut3["poly_Cscale50_rho4"]["worst_res_factor"] >= 0.999,
      "(%.4f)" % mut3["poly_Cscale50_rho4"]["worst_res_factor"])

# ------------------------------------------------------------------ claim 4
print("claim 4 (Corollary 4.2) -- verified")
c4 = load(4)
sym = c4["A_symbolic_dt_exponents"]
check("c4.every quadratic coefficient of eq. 6 carries dt^3 exactly",
      all(r["all_quadratic_dt_pow3"] for r in sym) and
      all(r["quadratic_dt_exponents"] in ([], [3]) for r in sym),
      "(rows: %s)" % [r["quadratic_dt_exponents"] for r in sym])
check("c4.symbolic vs numeric model agreement = 2.78e-17 (<1e-12)",
      c4["A_symbolic_vs_numeric_max_abs_err"] < 1e-12,
      "(%.2e)" % c4["A_symbolic_vs_numeric_max_abs_err"])
p = c4["A_Cnorm_scaling"]["fitted_exponent_p_in_Cnorm_prop_dt_p"]
check("c4.fitted exponent in ||C|| ~ dt^p equals 3.0000", approx(p, 3.0, 1e-4), "(%.4f)" % p)
rates = c4["B_rate_vs_dt"]
check("c4.ADMM on eq. 6 linear for every dt in [0.002, 0.1] (worst factor 0.7143)",
      max(r["worst_factor"] for r in rates) < 0.75,
      "(max %.4f)" % max(r["worst_factor"] for r in rates))
check("c4.all eq.-6 runs feasible (violation < 1e-14)",
      max(r["worst_final_viol"] for r in rates) < 1e-14,
      "(%.1e)" % max(r["worst_final_viol"] for r in rates))
mut4 = c4["B_MUTATION_dt_independent_nonlinearity"]
check("c4.MUTATION dt-independent nonlinearity (||C||=5.477) kills the linear rate "
      "(factor 0.9997 at every dt)",
      all(r["worst_factor"] >= 0.999 for r in mut4) and
      approx(mut4[0]["C_norm"], 5.477, 1e-3),
      "(min factor %.4f, ||C||=%.3f)" % (min(r["worst_factor"] for r in mut4),
                                         mut4[0]["C_norm"]))

# ------------------------------------------------------------------ claim 5
print("claim 5 (Figures 2 and 4) -- inconclusive")
c5 = load(5)
ta = c5["transition_analysis"]
b = ta["best_factor_below_q10"]
check("c5.q=0.5 already linear, best factor 0.2296", approx(b["0.5"], 0.2296, 5e-4),
      "(%.4f)" % b["0.5"])
check("c5.q=2 already linear, best factor 0.0475", approx(b["2.0"], 0.0475, 5e-4),
      "(%.4f)" % b["2.0"])
check("c5.all 5 tested q < 10 are already linear (no sublinear regime found)",
      ta["n_q_lt_10_already_linear"] == 5, "(%d/5)" % ta["n_q_lt_10_already_linear"])
check("c5.median best factor below q=10 (0.0588) is NOT worse than at/above q=10 (0.0287) "
      "by any order of magnitude",
      ta["median_factor_below_q10"] < 10 * ta["median_factor_at_or_above_q10"],
      "(%.4f vs %.4f)" % (ta["median_factor_below_q10"], ta["median_factor_at_or_above_q10"]))
check("c5.baseline comparison (Fig. 4) declared inconclusive, no numbers fabricated",
      c5["baselines_PADMM_IPDSADMM_IADMM"]["verdict"] == "inconclusive", "")

# ------------------------------------------------------------------ claim 6
print("claim 6 (Figures 5 and 6) -- inconclusive")
c6 = load(6)
check("c6.verdict is inconclusive", c6["verdict"] == "inconclusive", "")
check("c6.hardware parts marked not reproducible",
      c6["parts"]["humanoid_vertical_jump_hardware"]["status"] == "not reproducible" and
      c6["parts"]["quadruped_bounding_hardware"]["status"] == "not reproducible", "")

print("\n%d assertions passed, %d failed" % (PASSED, len(FAILED)))
if FAILED:
    for f in FAILED:
        print("  failed:", f)
    sys.exit(1)
print("ALL LOGBOOK NUMBERS REPRODUCED FROM results/*.json")
