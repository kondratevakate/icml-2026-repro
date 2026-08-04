"""Reproducibility gate: re-asserts every number quoted in logbook.md against results/*.json.

Run: .venv/bin/python check_reproducibility.py
Exit code 0 = all assertions pass.
"""
import json, os, sys

R = lambda n: json.load(open(os.path.join("results", n)))
checks, failures = [], []


def chk(name, cond, detail=""):
    checks.append((name, bool(cond), detail))
    if not cond:
        failures.append(f"{name}: {detail}")


def close(a, b, tol):
    return abs(a - b) <= tol


# ---------------- Claim 1 ----------------
c1 = R("claim1.json")
chk("c1.symbolic p(1-alpha')=1-alpha", c1["symbolic_identity_holds"])
chk("c1.exact PT coverage == 1-alpha on whole (alpha,p) grid",
    c1["exact_grid_max_abs_coverage_error"] == 0.0, str(c1["exact_grid_max_abs_coverage_error"]))
chk("c1.mutation (no alpha' adjust) loses up to 0.085 coverage",
    close(c1["mutated_grid_max_coverage_deficit"], 0.085, 1e-9),
    str(c1["mutated_grid_max_coverage_deficit"]))
chk("c1.MC: PT coverage >= nominal - 2sem for all (alpha,p)",
    c1["summary"]["all_mc_pt_coverage_ge_nominal_minus_2sem"])
chk("c1.MC: PT shorter than VCP for all (alpha,p)", c1["summary"]["all_mc_pt_shorter_than_vcp"])
m = c1["mc_correct"]["alpha0.1_p0.96"]
chk("c1.MC alpha=.1,p=.96 VCP len 22.704", close(m["vcp_length"]["mean"], 22.704, 5e-3))
chk("c1.MC alpha=.1,p=.96 PT len 22.294", close(m["pt_length"]["mean"], 22.294, 5e-3))
chk("c1.MC alpha=.1,p=.96 PT coverage 0.9006", close(m["pt_coverage"]["mean"], 0.9006, 5e-4))
mm = c1["mc_mutation_no_alpha_adjust"]["alpha0.1_p0.96"]
chk("c1.MUT1 coverage collapses to 0.8663 (< 0.90)",
    close(mm["pt_coverage"]["mean"], 0.8663, 5e-4) and mm["pt_coverage"]["mean"] < 0.90)
mg = c1["mc_mutation_gaussian_wellspecified"]["alpha0.1_p0.96"]
chk("c1.MUT2 well-specified Gaussian: PT LONGER (3.594 > 3.312)",
    mg["pt_length"]["mean"] > mg["vcp_length"]["mean"]
    and close(mg["pt_length"]["mean"], 3.594, 5e-3) and close(mg["vcp_length"]["mean"], 3.312, 5e-3))
chk("c1.Example 3 inequality holds on all sampled (alpha,p)", c1["example3_inequality_all_hold"])

# ---------------- Claim 2 ----------------
c2 = R("claim2.json")
chk("c2.all 16 swept configs give PT shorter than VCP", c2["all_configs_pt_shorter"])
best = c2["best_matching_config"]
chk("c2.best config = component mean 10, n=1000",
    best["component_mean"] == 10.0 and best["n_per_fold"] == 1000, str(best["component_mean"]))
chk("c2.best VCP len 22.837 vs paper 22.894", close(best["vcp_len"][0], 22.837, 5e-3))
chk("c2.best PT len 22.489 vs paper 22.614", close(best["pt_len"][0], 22.489, 5e-3))
chk("c2.best VCP len within 0.06 of paper 22.894", best["abs_err_vcp_len"] < 0.06)
chk("c2.best PT len within 0.13 of paper 22.614", best["abs_err_pt_len"] < 0.13)
lit = [r for r in c2["sweep"] if r["component_mean"] == 20.0 and r["n_per_fold"] == 1000][0]
chk("c2.literal mu=20 reading gives ~43.5 -> 42.5 (2x paper scale)",
    close(lit["vcp_len"][0], 43.520, 5e-3) and close(lit["pt_len"][0], 42.547, 5e-3))

# ---------------- Claim 4 ----------------
c4 = R("claim4.json")
chk("c4.symbolic Var == p(1-p)L^2 (Prop. 2)", c4["symbolic_matches_prop2"])
chk("c4.empirical IS matches Prop.2 closed form within 2.5%",
    c4["max_rel_err_vs_prop2"] < 0.025, str(c4["max_rel_err_vs_prop2"]))
chk("c4.IS > 0 for every p in the grid", c4["all_IS_positive"] and c4["min_IS_empirical"] > 4.9)
chk("c4.deterministic VCP has IS == 0 (< 1e-20)", c4["max_IS_vcp"] < 1e-20, str(c4["max_IS_vcp"]))
chk("c4.MUTATION p=1 -> IS == 0 (< 1e-20)", c4["mutation_max_IS"] < 1e-20, str(c4["mutation_max_IS"]))

# ---------------- Claim 5 ----------------
c5 = R("claim5.json")
chk("c5.localized-CP length == PT length up to finite-sample quantile index (<0.41)",
    c5["max_length_diff_vs_PT_eps_le_1e-9"] < 0.41, str(c5["max_length_diff_vs_PT_eps_le_1e-9"]))
chk("c5.Q_localized within 0.21 of base quantile at alpha'",
    c5["max_Q_gap_vs_base_alpha_prime_eps_le_1e-9"] < 0.21,
    str(c5["max_Q_gap_vs_base_alpha_prime_eps_le_1e-9"]))
chk("c5.randomized localized CP is shorter than VCP on every seed/p",
    c5["all_localized_shorter_than_vcp"])
chk("c5.IS flags it: IS >= 10.0 on every seed/p", c5["min_IS_random_sigma"] >= 10.0,
    str(c5["min_IS_random_sigma"]))
chk("c5.MUTATION fixed sigma == 1: IS == 0 and length == VCP length",
    c5["max_IS_fixed_sigma"] < 1e-20 and c5["all_mutation_fixed_sigma_equals_vcp"])

# ---------------- Claim 3 (partial: 6 of 10 datasets) ----------------
if os.path.exists("results/claim3.json"):
    c3 = R("claim3.json")
    chk("c3.6 downloadable datasets attempted, 4 gated ones skipped",
        c3["n_datasets_attempted"] == 6 and len(c3["datasets_not_attempted"]) == 4)
    chk("c3.marginal coverage within 0.02 of 0.90 everywhere",
        c3["all_coverage_within_0.02_of_nominal"])
    chk("c3.4 of 6 attempted datasets shorter under PT", c3["n_pt_shorter"] == 4,
        f"n_pt_shorter={c3['n_pt_shorter']}/6")
    exp = {"bike": (20.98, 20.24), "bio": (21.46, 20.66), "concrete": (10.55, 10.21),
           "blog-data": (43.12, 42.72), "facebook-1": (21.98, 22.67), "facebook-2": (22.54, 23.56)}
    byname = {r["dataset"]: r for r in c3["results"]}
    for d, (lv, lp) in exp.items():
        r = byname[d]
        chk(f"c3.{d} VCP len {lv}", close(r["vcp_len"][0], lv, 5e-3), f"{r['vcp_len'][0]:.3f}")
        chk(f"c3.{d} PT len {lp}", close(r["pt_len"][0], lp, 5e-3), f"{r['pt_len'][0]:.3f}")
    chk("c3.IS_pt spans 5.46..95.83",
        close(min(r["IS_pt"][0] for r in c3["results"]), 5.46, 5e-3)
        and close(max(r["IS_pt"][0] for r in c3["results"]), 95.83, 5e-3))
    chk("c3.IS of PT-VCP > 0 on every dataset", c3["min_IS_pt"] > 0, str(c3["min_IS_pt"]))
    for r in c3["results"]:
        chk(f"c3.{r['dataset']} VCP IS == 0",
            all(s["IS_vcp"] == 0.0 for s in r["per_seed"]))
else:
    chk("c3.results present", False, "results/claim3.json missing")

for n, ok, d in checks:
    print(("PASS " if ok else "FAIL ") + n + (f"  [{d}]" if d else ""))
print(f"\n{sum(1 for _, ok, _ in checks if ok)}/{len(checks)} checks passed")
sys.exit(1 if failures else 0)
