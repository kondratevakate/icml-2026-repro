"""check_reproducibility.py — reproducibility gate for logbook.md (arm1).

Re-asserts every number quoted in logbook.md against results/*.json.
Exit code 0 = all assertions pass.
"""
import json, sys

R = lambda n: json.load(open(f"results/{n}.json"))
fails = []


def chk(label, cond, got):
    print(("PASS  " if cond else "FAIL  ") + f"{label}  (got: {got})")
    if not cond:
        fails.append(label)


def approx(a, b, rel=0.02):
    return abs(a - b) <= rel * max(abs(b), 1e-12)


# ---------------- claim 1 ----------------
c1, c1b = R("claim1"), R("claim1b")
chk("c1b case2 instances == 9241", c1b["case2_instances"] == 9241, c1b["case2_instances"])
chk("c1b bound violations == 0", c1b["main_bound_violations"] == 0, c1b["main_bound_violations"])
chk("c1b worst ratio ~ 0.4003", approx(c1b["main_worst_ratio"], 0.4003), c1b["main_worst_ratio"])
chk("c1b sub-bound tested == 8866", c1b["subbound_intersection_gamma_tested"] == 8866,
    c1b["subbound_intersection_gamma_tested"])
chk("c1b sub-bound violations == 0", c1b["subbound_intersection_gamma_violations"] == 0,
    c1b["subbound_intersection_gamma_violations"])
chk("c1b MUTATION argmax violations == 3545",
    c1b["mutation_argmax_bound_violations"] == 3545, c1b["mutation_argmax_bound_violations"])
chk("c1b MUTATION worst ratio ~ 3.09e4", approx(c1b["mutation_argmax_worst_ratio"], 3.085e4, 0.05),
    c1b["mutation_argmax_worst_ratio"])
chk("c1 Eq.31 matrix-norm violations == 0 over 6000",
    c1["main"]["eq31_matrix_norm_violations"] == 0 and c1["main"]["n_instances"] == 6000,
    (c1["main"]["eq31_matrix_norm_violations"], c1["main"]["n_instances"]))

# ---------------- claim 2 ----------------
c2 = R("claim2")
chk("c2 max |A_g P_g - b_g| <= 1.9e-11",
    c2["A_consistency"]["max_abs_residual_Ag_Pg_minus_bg"] <= 1.9e-11,
    c2["A_consistency"]["max_abs_residual_Ag_Pg_minus_bg"])
B = c2["B_degeneracy"]
chk("c2 full-col-rank spread <= 2.9e-12", B["max_spread_when_full_col_rank"] <= 2.9e-12,
    B["max_spread_when_full_col_rank"])
chk("c2 rank-deficient min spread ~ 0.2062", approx(B["min_spread_when_rank_deficient"], 0.2062),
    B["min_spread_when_rank_deficient"])
chk("c2 rank-deficient median spread ~ 0.8511",
    approx(B["median_spread_when_rank_deficient"], 0.8511), B["median_spread_when_rank_deficient"])
chk("c2 n cases 565 / 1641", (B["n_full_col_rank_cases"], B["n_rank_deficient_cases"]) == (565, 1641),
    (B["n_full_col_rank_cases"], B["n_rank_deficient_cases"]))
C = c2["C_trainability"]
chk("c2 trained loss ~ 0.4580", approx(C["mean_loss_trained"], 0.4580), C["mean_loss_trained"])
chk("c2 MUTATION w=0 loss ~ 0.8964", approx(C["mean_loss_orthogonal"], 0.8964), C["mean_loss_orthogonal"])
chk("c2 trained better on 8/8 seeds", C["seeds_where_trained_strictly_better"] == 8,
    C["seeds_where_trained_strictly_better"])

# ---------------- claim 3 ----------------
s = R("claim3")["sweep"]
chk("c3 instances == 2160", s["n_instances"] == 2160, s["n_instances"])
chk("c3 CAffNet violations == 0", s["caffnet_violating_instances"] == 0, s["caffnet_violating_instances"])
chk("c3 CAffNet max violation <= 2.8e-14", s["caffnet_max_violation"] <= 2.8e-14, s["caffnet_max_violation"])
chk("c3 HardNet-like violations == 999", s["hardnet_like_violating_instances"] == 999,
    s["hardnet_like_violating_instances"])
chk("c3 HardNet-like max violation ~ 28.47", approx(s["hardnet_like_max_violation"], 28.467),
    s["hardnet_like_max_violation"])
chk("c3 MUTATION k=1 only failures == 506", s["truncated_k_eq_1_violating"] == 506,
    s["truncated_k_eq_1_violating"])
chk("c3 MUTATION k=min only failures == 194", s["truncated_k_eq_min_only_violating"] == 194,
    s["truncated_k_eq_min_only_violating"])
chk("c3 max selected cardinality == 4", s["max_cardinality_of_selected_gamma"] == 4,
    s["max_cardinality_of_selected_gamma"])
f1 = R("claim3")["fig1_example"]
chk("c3 Fig1 three distinct feasible outputs",
    {round(f1[k]["y"][0], 3) for k in ("w_zero", "w_left", "w_right")} == {1.5, -3.5, 6.5},
    [f1[k]["y"] for k in ("w_zero", "w_left", "w_right")])

# ---------------- claim 4 ----------------
c4 = R("claim4")
chk("c4 5 seeds x 50000 epochs, 3 arms",
    all(c4[k]["n_seeds"] == 5 and c4[k]["epochs"] == 50000
        for k in ("nn_soft", "caffnet_tf", "tf_soft_mutation")),
    [(k, c4[k]["n_seeds"], c4[k]["epochs"]) for k in ("nn_soft", "caffnet_tf", "tf_soft_mutation")])
chk("c4 NN mse ~ 1.728e-3", approx(c4["nn_soft"]["mse_mean"], 1.728e-3), c4["nn_soft"]["mse_mean"])
chk("c4 CAffNet-TF mse ~ 7.96e-4", approx(c4["caffnet_tf"]["mse_mean"], 7.96e-4), c4["caffnet_tf"]["mse_mean"])
chk("c4 measured reduction ~ 53.94% (paper 73.33%)",
    approx(c4["measured_reduction_pct"], 53.94), c4["measured_reduction_pct"])
chk("c4 CAffNet-TF viol max <= 1e-7", c4["caffnet_tf"]["viol_max"] <= 1e-7, c4["caffnet_tf"]["viol_max"])
chk("c4 NN viol max ~ 0.3336", approx(c4["nn_soft"]["viol_max"], 0.33357), c4["nn_soft"]["viol_max"])
chk("c4 NN viol pct ~ 6.41%", approx(c4["nn_soft"]["viol_pct_mean"], 6.41), c4["nn_soft"]["viol_pct_mean"])
chk("c4 MUTATION (no CAffine layer) viol max ~ 0.2546",
    approx(c4["tf_soft_mutation"]["viol_max"], 0.25457), c4["tf_soft_mutation"]["viol_max"])
chk("c4 MUTATION mse ~ 2.557e-3", approx(c4["tf_soft_mutation"]["mse_mean"], 2.557e-3),
    c4["tf_soft_mutation"]["mse_mean"])
chk("c4 CAffNet-TF beats NN on 4/5 seeds",
    sum(1 for a, b in zip([r["mse"] for r in c4["nn_soft"]["runs"]],
                          [r["mse"] for r in c4["caffnet_tf"]["runs"]]) if b < a) == 4,
    [round(r["mse"], 6) for r in c4["caffnet_tf"]["runs"]])

# ---------------- claim 5 ----------------
c5 = R("claim5")["results"]
chk("c5 soft collided 5/5", c5["soft"]["n_seeds_collided"] == 5, c5["soft"]["n_seeds_collided"])
chk("c5 hardnet-like collided 5/5", c5["hardnet_like"]["n_seeds_collided"] == 5,
    c5["hardnet_like"]["n_seeds_collided"])
chk("c5 caffnet collided 0/5", c5["caffnet"]["n_seeds_collided"] == 0, c5["caffnet"]["n_seeds_collided"])
chk("c5 soft max violation ~ 7.314", approx(c5["soft"]["max_constraint_violation"], 7.3144),
    c5["soft"]["max_constraint_violation"])
chk("c5 hardnet-like max violation ~ 1.167",
    approx(c5["hardnet_like"]["max_constraint_violation"], 1.1673),
    c5["hardnet_like"]["max_constraint_violation"])
chk("c5 caffnet max violation <= 1e-12", c5["caffnet"]["max_constraint_violation"] <= 1e-12,
    c5["caffnet"]["max_constraint_violation"])

print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: ' + '; '.join(fails)}")
sys.exit(1 if fails else 0)
