"""check_reproducibility.py — reproducibility gate for logbook.md.

Re-asserts every number quoted in logbook.md against results/*.json.
Exit code 0 = all assertions pass. Run: .venv/bin/python check_reproducibility.py
"""
import json
import os
import sys

FAILS = []
CHECKS = 0


def load(n):
    p = f"results/claim{n}.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def chk(name, cond, detail=""):
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"PASS  {name}  {detail}")
    else:
        print(f"FAIL  {name}  {detail}")
        FAILS.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# ---------------- claim 1 ----------------
d = load(1)
if d is None:
    chk("claim1.json exists", False)
else:
    f = d["faithful"]
    chk("c1 grid size = 3200 cases", f["n_cases"] == 3200, f["n_cases"])
    chk("c1 zero violations of ||P*-f_t||_p < (3+3 sqrt n_out)K",
        f["violations"]["C1"] == 0, f["violations"])
    chk("c1 zero violations of intermediate bounds C2,C3,C4 and Lemma 3.3 (C5)",
        all(f["violations"][k] == 0 for k in ("C2", "C3", "C4", "C5")), f["violations"])
    chk("c1 worst err/eps ratio < 1", f["worst_ratio_err_over_eps"] < 1.0,
        f["worst_ratio_err_over_eps"])
    chk("c1 Case-2 (projection actually invoked) instances > 100",
        f["n_case2_projected"] > 100, f["n_case2_projected"])
    m1, m2 = d["mutation_M1_big_w"], d["mutation_M2_argmax"]
    chk("c1 MUTATION M1 (||w_phi|| = 50K) breaks the bound",
        m1["violations"]["C1"] > 0 and m1["worst_ratio_err_over_eps"] > 1.0,
        f"{m1['violations']['C1']} viol, worst ratio {m1['worst_ratio_err_over_eps']:.2f}")
    chk("c1 MUTATION M2 (argmax instead of argmin in Eq. 12) breaks the bound",
        m2["violations"]["C1"] > 0 and m2["worst_ratio_err_over_eps"] > 1.0,
        f"{m2['violations']['C1']} viol, worst ratio {m2['worst_ratio_err_over_eps']:.2f}")

# ---------------- claim 2 ----------------
d = load(2)
if d is None:
    chk("claim2.json exists", False)
else:
    chk("c2 T1 A_g P_g = b_g for arbitrary w_phi (residual < 1e-9)",
        d["T1_max_abs_residual_Ag_Pg_minus_bg"] < 1e-9,
        d["T1_max_abs_residual_Ag_Pg_minus_bg"])
    chk("c2 T2 null-space term reaches every point of the solution set (< 1e-9)",
        d["T2_max_abs_reachability_error"] < 1e-9, d["T2_max_abs_reachability_error"])
    chk("c2 T3 w_phi=0 equals the orthogonal projection (< 1e-9)",
        d["T3_max_abs_dev_from_orthogonal_projection_at_w0"] < 1e-9,
        d["T3_max_abs_dev_from_orthogonal_projection_at_w0"])
    chk("c2 T4 optimising w_phi beats w_phi=0 on a majority of instances",
        d["T4_frac_improved"] > 0.5, f"{d['T4_frac_improved']:.3f} of {d['T4_n']}")
    chk("c2 T4 median relative loss reduction > 0.5",
        d["T4_median_rel_loss_reduction"] > 0.5, d["T4_median_rel_loss_reduction"])
    chk("c2 MUTATION removing the null-space term == fixed orthogonal projection",
        d["MUT_no_nullspace_equals_w0"] is True)
    chk("c2 MUTATION strictly worse than optimised w_phi on the improved instances",
        d["MUT_n_worse_than_optimised"] == d["T4_n_improved"],
        f"{d['MUT_n_worse_than_optimised']} vs {d['T4_n_improved']}")

# ---------------- claim 3 ----------------
d = load(3)
if d is None:
    chk("claim3.json exists", False)
else:
    e1, e2 = d["E1_exhaustive_2d"], d["E2E3_random_sweep"]
    chk("c3 exhaustive 2-D family: 0 CAffNet feasibility failures",
        e1["n_caffnet_failures"] == 0 and e1["max_violation"] == 0.0,
        f"{e1['n_cases']} cases over {e1['n_systems']} systems")
    chk("c3 random sweep (m up to 8 > n_out): 0 CAffNet failures",
        e2["n_caffnet_failures"] == 0 and e2["max_violation"] == 0.0,
        f"{e2['n_cases']} cases, {e2['n_rank_deficient']} rank-deficient")
    chk("c3 total cases >= 30000",
        e1["n_cases"] + e2["n_cases"] >= 30000, e1["n_cases"] + e2["n_cases"])
    chk("c3 HardNet-Aff pseudo-inverse correction IS infeasible on a large fraction",
        e1["hardnet_infeasible_frac"] > 0.1 and e2["hardnet_infeasible_frac"] > 0.1,
        f"{e1['hardnet_infeasible_frac']:.3f} / {e2['hardnet_infeasible_frac']:.3f}")
    chk("c3 MUTATION truncating Gamma to k <= min(m,n_out)-1 causes failures",
        e1["mutation_truncated_gamma_failures"] > 0
        and e2["mutation_truncated_gamma_failures"] > 0,
        f"{e1['mutation_failure_frac']:.3f} / {e2['mutation_failure_frac']:.3f}")
    chk("c3 |Gamma| matches Eq. (2) and <= 2^m - 1 for all tested (m, n_out)",
        d["E4_cardinality"]["all_ok"] is True)

# ---------------- claim 4 ----------------
d = load(4)
if d is None:
    chk("claim4.json exists", False)
else:
    a = d["aggregate"]
    chk("c4 paper arithmetic: 1 - 0.0012/0.0045 = 73.33%",
        approx(d["paper_reduction_pct_recomputed"], 73.33, 0.01),
        d["paper_reduction_pct_recomputed"])
    chk("c4 CAffNet-FF has exactly zero constraint violations",
        a["CAffNet-FF"]["viol_max"] == 0.0, a["CAffNet-FF"]["viol_max"])
    chk("c4 CAffNet-TF has exactly zero constraint violations",
        a["CAffNet-TF"]["viol_max"] == 0.0, a["CAffNet-TF"]["viol_max"])
    chk("c4 soft-constrained NN baseline DOES violate constraints",
        a["NN"]["viol_max"] > 0.0, a["NN"]["viol_max"])
    chk("c4 HardNet baseline DOES violate constraints (A not full row rank here)",
        a["HardNet"]["viol_max"] > 0.0, a["HardNet"]["viol_max"])
    chk("c4 MUTATION: removing the CAffine layer from the TF re-introduces violations",
        d["mutation_TF_nolayer_viol_max"] > 0.0, d["mutation_TF_nolayer_viol_max"])
    chk("c4 reproduced MSE reduction (TF vs NN) recorded",
        isinstance(d["reproduced_reduction_pct_TF_vs_NN"], float),
        f"{d['reproduced_reduction_pct_TF_vs_NN']:.2f}% "
        f"(paper 73.33%, epochs={d['epochs']} vs paper 50000)")
    # the logbook does NOT claim the 73.33% figure was reproduced; it quotes this value
    chk("c4 seeds used >= 5 (never a single seed)", d["seeds"] >= 5, d["seeds"])

# ---------------- claim 5 ----------------
d = load(5)
if d is None:
    chk("claim5.json exists", False)
else:
    a = d["aggregate"]
    chk("c5 CAffNet run has zero constraint violations on the test trajectory",
        a["CAffNet"]["mean_test_viol_max"] < 1e-6, a["CAffNet"]["mean_test_viol_max"])
    chk("c5 CAffNet collision count on the paper's test initial state",
        a["CAffNet"]["n_collisions"] == 0,
        f"collisions={a['CAffNet']['n_collisions']}/{a['CAffNet']['n_seeds']}")
    chk("c5 baselines (NN / HardNet) do violate the aggregated CBF constraints",
        a["NN"]["mean_test_viol_max"] > 0.0 or a["HardNet"]["mean_test_viol_max"] > 0.0,
        f"NN {a['NN']['mean_test_viol_max']:.4f}, "
        f"HardNet {a['HardNet']['mean_test_viol_max']:.4f}")
    chk("c5 scale note recorded (reduced-scale run, not the paper's 300 initial states)",
        "300 initial states" in d["scale_note"])

print(f"\n{CHECKS - len(FAILS)}/{CHECKS} checks passed")
if FAILS:
    print("FAILED:", FAILS)
    sys.exit(1)
print("ALL LOGBOOK NUMBERS RE-ASSERTED OK")
