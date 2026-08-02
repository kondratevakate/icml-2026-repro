"""check_reproducibility.py — reproducibility gate for logbook.md.

Re-asserts every number quoted in logbook.md against results/claim*.json.
Run:  .venv/bin/python check_reproducibility.py     (exit 0 = all assertions pass)
"""
import json, sys

R = lambda n: json.load(open(f"results/claim{n}.json"))
fails, checks = [], 0


def chk(label, cond):
    global checks
    checks += 1
    if not cond:
        fails.append(label)


def close(a, b, tol=1e-4):
    return abs(float(a) - float(b)) <= tol


# ---------------- claim 1
c1 = R(1)
chk("c1 exponent symbolic 2*alpha + 1/2", c1["exponent_rbf_symbolic"] == "2*alpha + 1/2")
chk("c1 threshold 1/4", close(c1["alpha_threshold_rbf_float"][0], 0.25, 1e-12))
chk("c1 intermediate T_c^2 sqrt(T) matches", c1["intermediate_matches_bound"] is True)
for al, ex in (("1/10", 0.6999), ("1/5", 0.9000), ("1/4", 1.0000), ("1/3", 1.1667), ("1/2", 1.5000)):
    chk(f"c1 exponent at alpha={al} = {ex}", close(c1["exponent_grid"][al]["exponent"], ex, 1e-3))
chk("c1 alpha=1/3 not sublinear", c1["exponent_grid"]["1/3"]["sublinear"] is False)
chk("c1 all 8 Matern/Table-1 entries agree", all(v["agree"] for v in c1["matern_table1"].values()))
chk("c1 Table1 eta=1/3 cases1_2 = 2/9", close(c1["matern_table1"]["eta=1/3|cases1_2"]["empirical_sup_alpha"], 0.222, 1e-3))
chk("c1 Table1 eta=1/2 case3 = 0", close(c1["matern_table1"]["eta=1/2|case3"]["empirical_sup_alpha"], 0.0, 1e-12))
chk("c1 mutation -> ideal 1/2", c1["mutation_drop_Psi"]["matches_paper_ideal"] and
    c1["mutation_drop_Psi"]["exponent"] == "alpha + 1/2" and c1["mutation_drop_Psi"]["verdict_changed"])
chk("c1 verdict verified", c1["claim_alpha_quarter_confirmed"] is True)

# ---------------- claim 2
c2 = R(2)
chk("c2 exponent symbolic 7*alpha/2 + 1/2", c2["exponent_rbf_symbolic"] == "7*alpha/2 + 1/2")
chk("c2 threshold 1/7", close(c2["alpha_threshold_rbf_float"][0], 1 / 7, 1e-12))
chk("c2 intermediate T_c^{7/2} sqrt(T) matches", c2["intermediate_matches_bound"] is True)
for al, ex in (("1/7", 1.0000), ("1/6", 1.0833), ("1/4", 1.3750), ("1/3", 1.6667)):
    chk(f"c2 exponent at alpha={al} = {ex}", close(c2["exponent_grid"][al]["exponent"], ex, 1e-3))
chk("c2 alpha=1/6 not sublinear", c2["exponent_grid"]["1/6"]["sublinear"] is False)
chk("c2 all 8 Matern/Table-1 entries agree", all(v["agree"] for v in c2["matern_table1"].values()))
chk("c2 Table1 eta=1/4 case3 ~ 1/12", close(c2["matern_table1"]["eta=1/4|case3"]["empirical_sup_alpha"], 0.08329, 1e-4))
chk("c2 mutation -> ideal 1/3", c2["mutation_drop_Psi"]["matches_paper_ideal"] and
    c2["mutation_drop_Psi"]["exponent"] == "3*alpha/2 + 1/2" and c2["mutation_drop_Psi"]["verdict_changed"])
chk("c2 verdict verified", c2["claim_alpha_seventh_confirmed"] is True)

# ---------------- claim 3
c3 = R(3)
chk("c3 1040 configurations", c3["n_configurations"] == 1040)
chk("c3 RCGP sup deviation 1.4212", close(c3["rcgp_sup_deviation_all_configs"], 1.4212, 1e-3))
chk("c3 RCGP sup dev / sigma_uc 1.5437", close(c3["rcgp_sup_deviation_over_sigma_uc"], 1.5437, 1e-3))
chk("c3 RCGP drift 1e6->1e12 = 9.93e-07", close(c3["rcgp_max_absolute_drift_1e6_to_1e12"], 9.932e-07, 1e-9))
chk("c3 GP sup deviation 4.983e+11", close(c3["gp_sup_deviation_all_configs"], 4.98309928919e11, 1e4))
chk("c3 GP growth ratio >= 999998", c3["gp_min_growth_ratio_1e6_to_1e12"] >= 999998.0)
chk("c3 RCGP bounded", c3["rcgp_bounded"] is True)
chk("c3 mutation deviation 4.536e+11", close(c3["mutation_constant_weight_sup_deviation"], 4.53585364497e11, 1e4))
chk("c3 mutation breaks boundedness", c3["mutation_breaks_boundedness"] is True)

# ---------------- claim 4
c4 = R(4)
chk("c4 Psi(0) = 1", close(c4["Psi_at_0"], 1.0, 1e-15))
chk("c4 FC bound collapses to GP-UCB rate", c4["symbolic_fc_equals_gpucb"] is True and
    c4["fc_bound_at_Tc0"] == "sqrt(T)*sqrt(betaprime)*sqrt(gamma)")
chk("c4 A2 bound collapses to GP-UCB rate", c4["symbolic_a2_equals_gpucb"] is True and
    c4["a2_bound_at_Tc0"] == "sqrt(T)*sqrt(betaprime)*sqrt(gamma)")
chk("c4 FC query diff 0.0", c4["max_query_diff_fc"] == 0.0)
chk("c4 A2 query diff 0.0", c4["max_query_diff_a2"] == 0.0)
chk("c4 FC regret diff 0.0", c4["max_abs_regret_diff_fc_vs_gpucb"] == 0.0)
chk("c4 A2 regret diff 0.0", c4["max_abs_regret_diff_a2_vs_gpucb"] == 0.0)
for k in ("gp_ucb_cum_regret", "fc_cum_regret", "a2_cum_regret"):
    chk(f"c4 mean cum regret {k} = 34.9909", close(c4["mean_cum_regret"][k], 34.9909, 1e-3))
chk("c4 mutation diverges on 10/10 seeds", c4["mutation_n_seeds_with_diverging_trajectory"] == 10)
chk("c4 mutation max regret diff 34.7054", close(c4["mutation_max_regret_diff"], 34.7054, 1e-3))

# ---------------- claim 5
c5 = R(5)
A, B = c5["A_maintext_T100"], c5["B_appendix_T30"]
chk("c5 setting A: T=100, T_c=5, near 0.2, far 0.5",
    A["n_iter"] == 100 and A["T_c_budget"] == 5 and A["near"] == 0.2 and A["far"] == 0.5)
chk("c5 setting B: T=30, T_c=4, near 0.1, far 0.4",
    B["n_iter"] == 30 and B["T_c_budget"] == 4 and B["near"] == 0.1 and B["far"] == 0.4)
expectA = {"fc_rcgp_ucb[Tc=zero,L=1.96]": (219.27, 27.40), "a2_rcgp_ucb[Tc=zero,L=1.96]": (224.62, 29.18),
           "gp_ucb": (247.37, 33.34), "studentt_ucb": (438.82, 17.58), "diagnostics_gp": (980.93, 60.80),
           "fc_rcgp_ucb[Tc=est,L=1.96]": (793.61, 2.41), "a2_rcgp_ucb[Tc=est,L=1.96]": (778.99, 3.82)}
expectB = {"fc_rcgp_ucb[Tc=zero,L=1.96]": (148.42, 8.10), "a2_rcgp_ucb[Tc=zero,L=1.96]": (151.47, 8.51),
           "gp_ucb": (157.59, 8.28), "studentt_ucb": (191.62, 2.99), "diagnostics_gp": (255.02, 17.95),
           "fc_rcgp_ucb[Tc=est,L=1.96]": (235.02, 3.26), "a2_rcgp_ucb[Tc=est,L=1.96]": (215.22, 6.03)}
for setting, exp, tag in ((A, expectA, "A"), (B, expectB, "B")):
    for k, (mean, sem) in exp.items():
        chk(f"c5 {tag} {k} mean {mean}", close(setting["cum_regret"][k]["mean"], mean, 5e-3))
        chk(f"c5 {tag} {k} sem {sem}", close(setting["cum_regret"][k]["sem"], sem, 5e-3))
    r = setting["cum_regret"]
    chk(f"c5 {tag} ordering RCGP(L=1.96,Tc=0) < GP-UCB < Student-t < DiagnosticsGP",
        max(r["fc_rcgp_ucb[Tc=zero,L=1.96]"]["mean"], r["a2_rcgp_ucb[Tc=zero,L=1.96]"]["mean"])
        < r["gp_ucb"]["mean"] < r["studentt_ucb"]["mean"] < r["diagnostics_gp"]["mean"])
    chk(f"c5 {tag} adaptive-Tc variant worse than GP-UCB",
        r["fc_rcgp_ucb[Tc=est,L=1.96]"]["mean"] > r["gp_ucb"]["mean"])
    chk(f"c5 {tag} best rcgp variant is fc(L=1.96,Tc=0)", setting["best_rcgp_variant"] == "fc_rcgp_ucb[Tc=zero,L=1.96]")
    chk(f"c5 {tag} not all rcgp variants beat all baselines", setting["all_rcgp_variants_beat_all_baselines"] is False)
    chk(f"c5 {tag} GP-UCB is NOT the worst baseline (paper says it is)", setting["gp_ucb_is_worst_baseline"] is False)
    for m in ("fc_rcgp_ucb", "a2_rcgp_ucb"):
        me = setting["mutation_effect"][m]
        chk(f"c5 {tag} mutation {m} lands exactly on GP-UCB", me["mutated_equals_gp_ucb"] is True)
        chk(f"c5 {tag} mutation {m} worsens regret", me["mutation_worsens_regret"] is True)
chk("c5 A improvement over GP-UCB is 11.4%", close(
    100 * (A["cum_regret"]["gp_ucb"]["mean"] - A["cum_regret"]["fc_rcgp_ucb[Tc=zero,L=1.96]"]["mean"])
    / A["cum_regret"]["gp_ucb"]["mean"], 11.4, 0.05))
chk("c5 B improvement over GP-UCB is 5.8%", close(
    100 * (B["cum_regret"]["gp_ucb"]["mean"] - B["cum_regret"]["fc_rcgp_ucb[Tc=zero,L=1.96]"]["mean"])
    / B["cum_regret"]["gp_ucb"]["mean"], 5.8, 0.05))
chk("c5 A gap to GP-UCB within one s.e.",
    A["cum_regret"]["gp_ucb"]["mean"] - A["cum_regret"]["fc_rcgp_ucb[Tc=zero,L=1.96]"]["mean"]
    < A["cum_regret"]["gp_ucb"]["sem"])
u = c5["uncorrupted_T30"]["cum_regret"]
for k, v in (("a2_rcgp_ucb[Tc=zero,L=q95]", 34.70), ("gp_ucb", 34.99),
             ("fc_rcgp_ucb[Tc=zero,L=1.96]", 40.90), ("studentt_ucb", 42.29), ("diagnostics_gp", 233.43)):
    chk(f"c5 uncorrupted {k} = {v}", close(u[k]["mean"], v, 5e-3))
chk("c5 uncorrupted: RCGP ~ GP-UCB and better than Student-t/DiagnosticsGP",
    u["a2_rcgp_ucb[Tc=zero,L=q95]"]["mean"] < u["gp_ucb"]["mean"] < u["studentt_ucb"]["mean"] < u["diagnostics_gp"]["mean"])

print(f"checks run : {checks}")
print(f"failures   : {len(fails)}")
for f in fails:
    print("  FAIL:", f)
print("RESULT     :", "PASS" if not fails else "FAIL")
sys.exit(0 if not fails else 1)
