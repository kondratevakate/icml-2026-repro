"""Reproducibility gate: re-assert every number quoted in logbook.md against results/*.json.

Run: ./.venv/bin/python check_reproducibility.py
Exit code 0 = all assertions pass.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda n: json.load(open(os.path.join(HERE, "results", f"claim{n}.json")))
fails = []


def eq(name, got, want, tol=0.0):
    ok = (got == want) if tol == 0 else abs(got - want) <= tol
    print(f"{'OK ' if ok else 'FAIL'} {name}: {got!r} (expected {want!r})")
    if not ok:
        fails.append(name)


# ---- claim 1 ----
c1 = R(1)
eq("c1.symbolic_constant_closes", c1["symbolic_constant_closes"], True)
eq("c1.infeasible_instances", c1["infeasible_instances"], 0)
eq("c1.max_ratio_overall", c1["max_ratio_overall"], 1.686931229412946, 1e-12)
eq("c1.n_configs_bound_holds", c1["n_configs_bound_holds"], 8)
eq("c1.n_configs_total", c1["n_configs_total"], 8)
for (n_out, m), mx in zip([(1, 2), (1, 4), (2, 3), (2, 5), (3, 4), (3, 7), (4, 6), (5, 8)],
                          [0.9971001586566164, 0.9923570959997907, 1.6424769212615287,
                           1.446907792078362, 1.6027273263822814, 1.2320821394200527,
                           1.68252632685596, 1.686931229412946]):
    row = [r for r in c1["per_config"] if (r["n_out"], r["m"]) == (n_out, m)][0]
    eq(f"c1.max_ratio[{n_out},{m}]", row["max_ratio"], mx, 1e-12)
    eq(f"c1.holds[{n_out},{m}]", row["holds"], True)
for row, mx in zip(c1["mutation_worst_candidate"],
                   [18.96320931916251, 7.846957574033642, 112.1564423072167,
                    126.41046987139862, 71.4357431773435, 275.42964969616213,
                    202.0805111475756, 677.5939335688797]):
    eq(f"c1.mutation_max_ratio[{row['n_out']},{row['m']}]", row["max_ratio"], mx, 1e-9)
    eq(f"c1.mutation_bound_broken[{row['n_out']},{row['m']}]", row["holds"], False)

# ---- claim 2 ----
c2 = R(2)
eq("c2.T1_max_residual", c2["T1_max_subconstraint_residual_over_all_w"],
   1.1834977442504169e-13, 1e-20)
eq("c2.T1_n_checks", c2["T1_n_checks"], 360)
eq("c2.T2_rank_deficient_cases", c2["T2_rank_deficient_cases"], 86)
eq("c2.T2_min_spread", c2["T2_min_output_spread_over_w"], 1.530619010628436, 1e-12)
eq("c2.T2_mean_spread", c2["T2_mean_output_spread_over_w"], 2.732280847454395, 1e-12)
eq("c2.T2_full_rank_zero_nullspace", c2["T2_full_rank_cases_with_zero_nullspace"], 34)
eq("c2.T3_obj_trainable_w", c2["T3_mean_obj_trainable_w"], 1.859256323957484, 1e-9)
eq("c2.T3_obj_w_zero", c2["T3_mean_obj_w_zero_orthogonal_projection"], 4.500871503391247, 1e-9)
eq("c2.T3_median_gap", c2["T3_median_gap"], 0.11952442114037204, 1e-9)
eq("c2.T3_max_gap", c2["T3_max_gap"], 26.87171091351931, 1e-9)
eq("c2.T3_frac_better", c2["T3_frac_instances_trainable_w_strictly_better"], 0.6, 1e-12)
eq("c2.MUTATION_w_zero_never_better", c2["MUTATION_w_forced_zero_is_never_better"], True)

# ---- claim 3 ----
c3 = R(3)
eq("c3.n_instances_total", c3["n_instances_total"], 2100)
eq("c3.no_feasible_candidate", c3["caffnet_instances_with_no_feasible_candidate"], 0)
eq("c3.violations", c3["caffnet_instances_with_constraint_violation"], 0)
eq("c3.max_violation", c3["caffnet_max_violation_over_all_instances"],
   1.1075584893660562e-12, 1e-18)
eq("c3.hardnet_undefined", c3["hardnet_instances_undefined_due_to_rank_deficiency"], 1961)
eq("c3.hardnet_violations", c3["hardnet_instances_with_constraint_violation"], 0)
eq("c3.mutation_failures", c3["MUTATION_kmax_minus_1_failure_count"], 478)
eq("c3.mutation_total", c3["MUTATION_kmax_minus_1_total"], 2100)
eq("c3.cardinality_identity", c3["T3_max_cardinality_equals_min_m_nout"], True)
eq("c3.count_identity", c3["T3_gamma_count_identity_and_2m_bound"], True)

# ---- claim 4 ----
c4 = R(4)
eq("c4.epochs", c4["epochs"], 20000)
eq("c4.seeds", c4["seeds"], 5)
a = c4["aggregate"]
eq("c4.NN.mse_mean", a["NN"]["mse_mean"], 0.001839826621879348, 1e-15)
eq("c4.NN.mse_std", a["NN"]["mse_std"], 0.0024093821342899414, 1e-15)
eq("c4.NN.viol_max_mean", a["NN"]["viol_max_mean"], 0.1705633378904027, 1e-12)
eq("c4.NN.seeds_with_violation", a["NN"]["n_seeds_with_any_violation"], 5)
eq("c4.FF.mse_mean", a["CAffNet-FF"]["mse_mean"], 0.002884547281501829, 1e-15)
eq("c4.FF.mse_std", a["CAffNet-FF"]["mse_std"], 0.0019290806990403236, 1e-15)
eq("c4.FF.viol_max_mean", a["CAffNet-FF"]["viol_max_mean"], 0.0)
eq("c4.FF.seeds_with_violation", a["CAffNet-FF"]["n_seeds_with_any_violation"], 0)
eq("c4.TF.mse_mean", a["CAffNet-TF"]["mse_mean"], 0.0009070503751830563, 1e-15)
eq("c4.TF.mse_std", a["CAffNet-TF"]["mse_std"], 0.0005683786123839912, 1e-15)
eq("c4.TF.viol_max_mean", a["CAffNet-TF"]["viol_max_mean"], 0.0)
eq("c4.TF.seeds_with_violation", a["CAffNet-TF"]["n_seeds_with_any_violation"], 0)
eq("c4.reduction_TF_vs_NN_pct", c4["reproduced_reduction_TF_vs_NN_pct"], 50.69913847335672, 1e-9)
eq("c4.reduction_FF_vs_NN_pct", c4["reproduced_reduction_FF_vs_NN_pct"], -56.783647284944635, 1e-9)
eq("c4.caffnet_zero_violations", c4["caffnet_zero_violations"], True)
eq("c4.nn_has_violations", c4["nn_has_violations"], True)
eq("c4.paper_reduction", c4["paper_table2"]["reduction_TF_vs_NN_pct"], 73.33)

# ---- claim 5 ----
c5 = R(5)
eq("c5.m_constraints", c5["m_constraints"], 13)
eq("c5.n_out", c5["n_out"], 2)
eq("c5.n_gammas", c5["n_gammas"], 91)
ag = c5["aggregate"]
eq("c5.NN-soft.mean_collisions", ag["NN-soft"]["mean_collisions"], 20.333333333333332, 1e-9)
eq("c5.NN-soft.mean_control_violations", ag["NN-soft"]["mean_control_violations"], 24.0, 1e-9)
eq("c5.NN-soft.mean_arrived", ag["NN-soft"]["mean_arrived"], 47.666666666666664, 1e-9)
eq("c5.CAff.mean_collisions", ag["CAffNet-FF"]["mean_collisions"], 1.0, 1e-12)
eq("c5.CAff.mean_control_violations", ag["CAffNet-FF"]["mean_control_violations"],
   0.3333333333333333, 1e-9)
eq("c5.CAff.mean_arrived", ag["CAffNet-FF"]["mean_arrived"], 15.666666666666666, 1e-9)
eq("c5.apost.mean_collisions", ag["CAffNet-FF-aposteriori"]["mean_collisions"], 1.0, 1e-12)
eq("c5.apost.mean_arrived", ag["CAffNet-FF-aposteriori"]["mean_arrived"],
   13.333333333333334, 1e-9)
for r in c5["runs"]:
    if r["method"] == "NN-soft":
        eq(f"c5.NN-soft.seed{r['seed']}.infeasible_steps", r["infeasible_steps"], 0)
    if r["method"] == "CAffNet-FF":
        eq(f"c5.CAff.seed{r['seed']}.infeasible_steps", r["infeasible_steps"], 4)
        eq(f"c5.CAff.seed{r['seed']}.collisions", r["collisions"], 1)

print()
if fails:
    print(f"FAILED {len(fails)} assertions: {fails}")
    sys.exit(1)
print("ALL LOGBOOK NUMBERS RE-ASSERTED OK")
