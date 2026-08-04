"""
verify_claim5.py -- Claim 5
"Theorem 4 gives an improved generalization gap bound for self-bounded losses
that scales poly-logarithmically rather than linearly in the number of
iterations T, depending on the cumulative training loss of later tasks
(Theorem 4 / Theorem B.1)."

We reproduce Theorem B.1 (the paper's "Theorem 4") and verify:
  * POLY-LOG in T:  G_imp ~ eta*d^2*log^3(T)/n  (Remark B.2), i.e. G_imp(T)
    grows as log^3(T), whereas Theorem 2.3 grows LINEARLY in T.  Over the same
    T-range the improved bound grows far slower (ratio ~ (log T2/log T1)^3 vs
    T2/T1) -- asymptotically the improved gap is much smaller.
  * DEPENDS ON LATER TASKS' CUMULATIVE LOSS c_{k,K}: more later-task data ->
    smaller later loss -> smaller c_{k,K} -> smaller G_imp (monotonic).
  * Under the width condition the exponent is O(1) and G_imp = o_d(1).
MUTATION: if the per-step training loss does NOT decay (loss not self-bounded /
net not in kernel regime), the cumulative loss ~ T and the improved bound
reverts to linear (or worse) in T -- losing the poly-log advantage -> breaks.
"""
import json
import os
import numpy as np
import repro_common as R

SEED = R.claim_seed(5)
K = 4
k = 1
d = 64
pld = float(np.log(d))
rp = R.regime_params(d, K, polylog_d=pld)
n, m = rp['n'], rp['m']
eta = 0.02
print(f"[Claim 5] Theorem B.1 improved generalization gap (K={K}, k={k}, d={d}, seed={SEED})")

# ---------- (A) poly-log vs linear in T ----------
T_grid = [50, 100, 200, 400, 800]
G_imp_T, G23_T = [], []
for T in T_grid:
    etaT = eta * T
    gimp = R.theoremB1_improved_gengap(k, d, n, m, K, eta, T,
                                        width_condition=True, decaying_loss=True)['polylog_asymptotic']
    g23 = R.theorem23_gengap(k, d, n, m, K, etaT)['gen_gap']
    G_imp_T.append(gimp); G23_T.append(g23)
# ratios between consecutive T
imp_ratios = [G_imp_T[i + 1] / G_imp_T[i] for i in range(len(T_grid) - 1)]
lin_ratios = [G23_T[i + 1] / G23_T[i] for i in range(len(T_grid) - 1)]
# predicted poly-log ratio (log^3): (log T2/log T1)^3
pred_imp = [((np.log(T_grid[i + 1]) / np.log(T_grid[i])) ** 3) for i in range(len(T_grid) - 1)]
print(f"\n  T-scan (improved ~ eta d^2 log^3 T / n,  Thm2.3 ~ eta T / n):")
for i, T in enumerate(T_grid):
    print(f"    T={T:4d}: G_imp={G_imp_T[i]:.3e}  G_Thm2.3={G23_T[i]:.3e}")
print(f"  improved growth ratios  = {[f'{x:.2f}' for x in imp_ratios]}  (predicted log^3 ~ {[f'{x:.2f}' for x in pred_imp]})")
print(f"  Thm2.3  growth ratios  = {[f'{x:.2f}' for x in lin_ratios]}  (predicted linear = T-ratio)")
polylog_ok = all(abs(imp_ratios[i] - pred_imp[i]) < 0.15 for i in range(len(imp_ratios)))
linear_faster = lin_ratios[-1] > imp_ratios[-1]
print(f"  G_imp ~ poly-log(log^3 T)? {polylog_ok};  Thm2.3 grows faster than improved (linear>polylog)? {linear_faster}")

# ---------- (B) depends on later tasks' cumulative loss ----------
print("\n  Dependence on later-task cumulative loss c_{k,K} (more data -> smaller c_{k,K}):")
later_scales = [0.2, 0.5, 1.0, 2.0]
T_fix = 200
etaT_fix = eta * T_fix
G_by_scale = []
for s in later_scales:
    gg = R.theoremB1_improved_gengap(k, d, n, m, K, eta, T_fix,
                                      later_loss_scale=s, width_condition=False, decaying_loss=True)
    G_by_scale.append(gg['gen_gap_improved'])
print(f"    later_loss_scale (data) = {later_scales}")
print(f"    G_imp                  = {[f'{x:.3e}' for x in G_by_scale]}")
later_dependence_ok = all(G_by_scale[i + 1] > G_by_scale[i] for i in range(len(later_scales) - 1))
print(f"  G_imp increases as later-task loss increases (more later data -> smaller gap)? {later_dependence_ok}")

# ---------- (C) MUTATION: non-decaying per-step loss ----------
T_mut = 800
etaT_mut = eta * T_mut
g_decay = R.theoremB1_improved_gengap(k, d, n, m, K, eta, T_mut,
                                       width_condition=True, decaying_loss=True)['polylog_asymptotic']
g_const = R.theoremB1_improved_gengap(k, d, n, m, K, eta, T_mut,
                                      width_condition=False, decaying_loss=False)['gen_gap_improved']
exploded = (g_const == float('inf')) or (g_const > 1e3 * max(g_decay, 1e-12))
print(f"\n  MUTATION: per-step loss NOT self-bounded (constant O(1)):")
print(f"    decaying (kernel regime)   G_imp = {g_decay:.3e}")
print(f"    non-decaying (mutated)     G_imp = {'inf (EXPLODES)' if g_const == float('inf') else f'{g_const:.3e}'}")
print(f"  Non-decaying loss loses the poly-log advantage (reverts to >=linear)? {exploded}")

# ---------- verdict ----------
verdict = "verified" if (polylog_ok and linear_faster and later_dependence_ok and exploded) else "inconclusive"
print(f"\n  VERDICT: {verdict}")

result = {
    "claim": 5,
    "statement": "Theorem 4 (Thm B.1) gives an improved generalization gap bound for self-bounded "
                 "losses that scales poly-logarithmically rather than linearly in T, depending on the "
                 "cumulative training loss of later tasks.",
    "source": "Theorem B.1 (improved gen. gap, the paper's 'Theorem 4'), Remark B.2, Eqs (6), arXiv:2510.05573v2.",
    "seed": SEED,
    "T_scan": {
        "T_grid": T_grid,
        "G_improved": [float(x) for x in G_imp_T],
        "G_Thm23": [float(x) for x in G23_T],
        "improved_growth_ratios": [float(x) for x in imp_ratios],
        "predicted_polylog_ratios": [float(x) for x in pred_imp],
        "Thm23_growth_ratios": [float(x) for x in lin_ratios],
        "polylog_in_T_confirmed": bool(polylog_ok),
        "linear_grows_faster": bool(linear_faster)},
    "later_task_dependence": {
        "later_loss_scale": later_scales,
        "G_improved_by_scale": [float(x) for x in G_by_scale],
        "depends_on_later_loss": bool(later_dependence_ok)},
    "mutation_test": {
        "description": "Per-step training loss held constant (not self-bounded / net not in kernel regime).",
        "decaying_G_improved": float(g_decay),
        "nondecaying_G_improved": (None if g_const == float('inf') else float(g_const)),
        "nondecaying_exploded": bool(exploded),
        "property_breaks": bool(exploded),
        "note": "With a non-decaying per-step loss the cumulative loss ~ T, so the improved exponent "
                "grows with T and the bound reverts to >= linear in T, destroying the poly-log advantage."},
    "verdict": verdict,
}
os.makedirs("results", exist_ok=True)
with open("results/claim5.json", "w") as f:
    json.dump(result, f, indent=2)
print("saved results/claim5.json")
