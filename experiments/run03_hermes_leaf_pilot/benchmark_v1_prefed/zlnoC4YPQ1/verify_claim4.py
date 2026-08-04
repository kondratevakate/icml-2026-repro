"""verify_claim4.py — Claim 4: zero-cost robustness (Theorem 4.1, Sec 4.3).
With T_c = 0 both FC-RCGP-UCB and A2-RCGP-UCB recover the GP-UCB bound O(sqrt(T beta'_T gamma_T)).

Two independent pieces of evidence:
 (A) symbolic: substituting T_c = 0 into the Theorem 4.2 and Theorem 4.3 bounds (with Psi(0)=1)
     must yield EXACTLY sqrt(beta'_T T gamma_T);
 (B) exact numeric mechanism: when the plateau condition holds (all residuals |y-g| <= L), the
     P-IMQ weight equals W, hence J_w = I and m_w = 0, so the RCGP posterior is bit-identical to
     the GP posterior, the acquisition multipliers coincide (Psi(0)=1, beta_t = beta'_t) and the
     algorithms produce the IDENTICAL query sequence and identical cumulative regret as GP-UCB.
     Checked on the uncorrupted Forrester benchmark (Sec 5.3 / Fig 3 setting), 10 seeds, 30 iters.
MUTATION: shrink the plateau (L = 0.05, plateau condition violated) -> the trajectories must
diverge from GP-UCB and the regret must change.
"""
import json, numpy as np, sympy as sp
from bo import run_bo
from rcgp import psi

out = {"claim": 4, "source": "Theorem 4.1 (Sec 4.3), Appendix F; mechanism: Definition 3.1 / Sec 3",
       "command": ".venv/bin/python verify_claim4.py"}

# ---------------- (A) symbolic collapse of both bounds at T_c = 0
T, bp, gam = sp.symbols("T betaprime gamma", positive=True)
Tc = 0
Psi0 = sp.sqrt(1 + Tc * (1 + Tc))
fc = sp.simplify(Psi0 * (1 + sp.sqrt(Tc)) * sp.sqrt(bp * T * (gam + Tc)))
a2 = sp.simplify((1 + Tc * Psi0 ** 2) * sp.sqrt(bp * T * (gam + Tc)))
target = sp.sqrt(bp * T * gam)
out["Psi_at_0"] = float(psi(0))
out["fc_bound_at_Tc0"] = str(fc)
out["a2_bound_at_Tc0"] = str(a2)
out["symbolic_fc_equals_gpucb"] = bool(sp.simplify(fc - target) == 0)
out["symbolic_a2_equals_gpucb"] = bool(sp.simplify(a2 - target) == 0)

# ---------------- (B) exact numeric equivalence on the uncorrupted Forrester benchmark
SEEDS = list(range(10))
N_ITER = 30
per_seed = {}
for s in SEEDS:
    g = run_bo("gp_ucb", s, N_ITER, corrupt=False)
    f = run_bo("fc_rcgp_ucb", s, N_ITER, corrupt=False, fixed_L=10.0, tc_mode="zero")
    a = run_bo("a2_rcgp_ucb", s, N_ITER, corrupt=False, fixed_L=10.0, tc_mode="zero")
    per_seed[s] = {
        "gp_ucb_cum_regret": g["cum_regret"],
        "fc_cum_regret": f["cum_regret"],
        "a2_cum_regret": a["cum_regret"],
        "max_query_diff_fc": float(np.max(np.abs(np.array(g["queries"]) - np.array(f["queries"])))),
        "max_query_diff_a2": float(np.max(np.abs(np.array(g["queries"]) - np.array(a["queries"])))),
        "abs_regret_diff_fc": abs(g["cum_regret"] - f["cum_regret"]),
        "abs_regret_diff_a2": abs(g["cum_regret"] - a["cum_regret"]),
    }
out["per_seed"] = per_seed
out["max_abs_regret_diff_fc_vs_gpucb"] = max(v["abs_regret_diff_fc"] for v in per_seed.values())
out["max_abs_regret_diff_a2_vs_gpucb"] = max(v["abs_regret_diff_a2"] for v in per_seed.values())
out["max_query_diff_fc"] = max(v["max_query_diff_fc"] for v in per_seed.values())
out["max_query_diff_a2"] = max(v["max_query_diff_a2"] for v in per_seed.values())
out["mean_cum_regret"] = {
    k: float(np.mean([per_seed[s][k] for s in SEEDS]))
    for k in ("gp_ucb_cum_regret", "fc_cum_regret", "a2_cum_regret")}
out["identical_trajectories_all_seeds"] = bool(out["max_query_diff_fc"] == 0.0
                                               and out["max_query_diff_a2"] == 0.0)

# ---------------- MUTATION: violate the plateau condition (tiny L)
mut = {}
for s in SEEDS:
    g = run_bo("gp_ucb", s, N_ITER, corrupt=False)
    f = run_bo("fc_rcgp_ucb", s, N_ITER, corrupt=False, fixed_L=0.05, tc_mode="zero")
    mut[s] = {"abs_regret_diff": abs(g["cum_regret"] - f["cum_regret"]),
              "max_query_diff": float(np.max(np.abs(np.array(g["queries"]) - np.array(f["queries"]))))}
out["mutation_small_L"] = mut
out["mutation_n_seeds_with_diverging_trajectory"] = sum(
    1 for v in mut.values() if v["max_query_diff"] > 0)
out["mutation_max_regret_diff"] = max(v["abs_regret_diff"] for v in mut.values())
out["mutation_breaks_equivalence"] = bool(out["mutation_n_seeds_with_diverging_trajectory"] >= 5)

out["verdict_ok"] = bool(out["symbolic_fc_equals_gpucb"] and out["symbolic_a2_equals_gpucb"]
                         and out["identical_trajectories_all_seeds"]
                         and out["mutation_breaks_equivalence"])
json.dump(out, open("results/claim4.json", "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if k not in ("per_seed", "mutation_small_L")}, indent=2))
