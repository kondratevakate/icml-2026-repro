"""check_reproducibility.py — reproducibility gate for logbook.md.

Re-asserts every number quoted in logbook.md against results/*.json.
Exit code 0 = all assertions pass.

Run: .venv/bin/python check_reproducibility.py
"""
import json, sys, math

FAIL = []
def load(n):
    return json.load(open(f"results/claim{n}.json"))

def chk(cond, msg):
    if cond:
        print(f"  PASS  {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAIL.append(msg)

def approx(a, b, rel=1e-9):
    return math.isclose(a, b, rel_tol=rel)

# ---------------- claim 1 ----------------
print("claim 1 (Proposition 1, discrete conservation)")
c1 = load(1)
chk(c1["verdict"] == "verified", "verdict == verified")
chk(c1["symbolic_exact"] is True, "symbolic sum difference is identically 0 for N=3,4,5,6")
chk(all(v == "0" for v in c1["symbolic_sum_difference"].values()), "all symbolic residuals '0'")
f64 = c1["conservation_rel_err_float64"]; f32 = c1["conservation_rel_err_float32"]
chk(f64["n_configs"] == 240 and f32["n_configs"] == 240, "240 configs per dtype (4 shapes x 3 radii x 20 seeds)")
chk(f64["max"] < 1e-13, f"float64 max rel conservation error {f64['max']:.3e} < 1e-13")
chk(approx(f64["max"], 4.436477518062519e-16), "float64 max == 4.436e-16 (logbook)")
chk(approx(f32["max"], 7.392088142433329e-08), "float32 max == 7.392e-08 (logbook)")
chk(approx(f32["median"], 1.4752693137761787e-08), "float32 median == 1.475e-08 (logbook)")
chk(1e-8 <= f32["max"] <= 3e-7, "float32 max lies in the paper's quoted 1e-7..1e-8 band")
m = c1["mutation"]
chk(approx(m["asymmetric_inflow"]["min"], 0.009676584590255922), "mutation asymmetric_inflow min == 9.677e-3")
chk(approx(m["nonperiodic"]["min"], 0.02259045063778126), "mutation nonperiodic min == 2.259e-2")
chk(c1["mutation_ratio_asymmetric_over_baseline"] > 1e4, "asymmetric mutation >1e4x baseline error")
chk(c1["mutation_ratio_nonperiodic_over_baseline"] > 1e4, "nonperiodic mutation >1e4x baseline error")

# ---------------- claim 2 ----------------
print("claim 2 (Propositions 2 & 3, L-head / U-head bound guarantees)")
c2 = load(2)
chk(c2["verdict"] == "verified", "verdict == verified")
chk(c2["symbolic_L_margin"] == "a*(1 - alpha)", "symbolic L margin == a*(1-alpha)")
chk(c2["symbolic_U_margin"] == "b*(1 - beta)", "symbolic U margin == b*(1-beta)")
chk(c2["n_configs"] == 540, "540 configs (4 shapes x 3 radii x 15 seeds x 3 logit scales)")
chk(c2["L_head_n_violating_configs"] == 0, "L-head: 0/540 configs violate the lower bound")
chk(c2["U_head_n_violating_configs"] == 0, "U-head: 0/540 configs violate the upper bound")
chk(c2["L_head_max_lower_bound_violation"] < 1e-12,
    f"L-head worst excursion {c2['L_head_max_lower_bound_violation']:.3e} < 1e-12 (float roundoff)")
chk(c2["U_head_max_upper_bound_violation"] < 1e-12,
    f"U-head worst excursion {c2['U_head_max_upper_bound_violation']:.3e} < 1e-12 (float roundoff)")
chk(c2["no_clipping_in_implementation"] is True, "no clipping/projection in the implementation")
mL = c2["mutation_alpha_cap_1.6"]; mU = c2["mutation_beta_cap_1.6"]
chk(mL["n_violating"] == mL["n_configs"] == 120, "mutation alpha_cap=1.6: 120/120 configs violate")
chk(mU["n_violating"] == mU["n_configs"] == 120, "mutation beta_cap=1.6: 120/120 configs violate")
chk(mL["max_violation"] > 1.0 and mU["max_violation"] > 1.0, "mutation violations are O(1)+, not roundoff")

# ---------------- claim 6 ----------------
print("claim 6 (D-head: DCL, not architecture, enforces dual bounds)")
c6 = load(6)
chk(c6["verdict"] == "verified", "verdict == verified")
chk(c6["d_head_max_conservation_rel_err_float64"] < 1e-13,
    f"D-head averaged update conserves: {c6['d_head_max_conservation_rel_err_float64']:.3e} < 1e-13")
ut = c6["untrained_branches"]; af = c6["after_DCL_minimization"]; mu = c6["mutation_maximize_DCL"]
chk(ut["n_seeds_with_violation"] == ut["n_seeds"] == 6, "6/6 untrained seeds show bound violations")
chk(ut["lb_rate"] > 1.0 and ut["ub_rate"] > 1.0,
    f"untrained violation rates lb {ut['lb_rate']:.2f}% / ub {ut['ub_rate']:.2f}% are percent-scale")
chk(ut["max_mag"] > 0.4, f"untrained max violation magnitude {ut['max_mag']:.3f} > 0.4")
chk(af["n_seeds_zero_violation"] == 6, "after DCL minimization: 6/6 seeds have exactly 0 violations")
chk(af["lb_rate"] == 0.0 and af["ub_rate"] == 0.0, "after DCL: violation rate exactly 0%")
chk(af["dcl"] < 1e-6 and af["dcl"] < 1e-6 * 1e0, f"after DCL: L_DCL {af['dcl']:.3e} < 1e-6")
chk(af["dcl"] < 0.01 * ut["dcl"], "DCL reduced by >100x versus untrained")
chk(mu["n_seeds_zero_violation"] == 0, "MUTATION (maximize DCL): 0/6 seeds reach zero violations")
chk(mu["max_mag"] > ut["max_mag"], f"MUTATION magnifies violations {mu['max_mag']:.3f} > {ut['max_mag']:.3f}")
ab = c6["analytic_bound_violation_le_half_branch_gap"]
chk(ab["holds_all_20_seeds"] is True, "violation <= 0.5*|du_out - du_in| holds on all 20 seeds")
sp = c6["dcl_vs_violation_spearman"]
chk(sp["rho"] > 0.7 and sp["n"] == 90,
    f"Spearman rho(sqrt(DCL), violation magnitude) = {sp['rho']:.3f} over n=90")

# ---------------- claims 3,4,5 ----------------
for n in (3, 4, 5):
    print(f"claim {n} (empirical table claim)")
    c = load(n)
    chk(c["verdict"] == "inconclusive", "verdict == inconclusive")
    chk(c["toy_substitute_used"] is False, "no toy substitute was produced")
    chk(len(c["blockers"]) >= 4, f"{len(c['blockers'])} concrete blockers recorded")
    chk(c["code_url_hits_in_paper_text"] in (0, None),
        "no code/dataset URL found in the paper text")
c4 = load(4)
q = c4["quoted_numbers_not_reproduced"]
chk(q["anchored_v_ub_fluxnet_pct"] == 1.87 and q["table4_v_ub_fluxnet_resnet_pct"] == 2.87,
    "claim-4 discrepancy recorded: anchored V_ub 1.87% vs Table 4's 2.87% (ResNet) / 2.01% (FNO)")

print()
if FAIL:
    print(f"REPRODUCIBILITY GATE: FAILED ({len(FAIL)} assertion(s))")
    for f in FAIL:
        print("  -", f)
    sys.exit(1)
print("REPRODUCIBILITY GATE: PASSED — every number quoted in logbook.md matches results/*.json")
