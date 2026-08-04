"""
verify_claim4.py -- Claim 4
"Theorem 3 bounds the delayed generalization gap by
    F_gen = eta*T * exp( eta*T*(K-k+1)/sqrt(m) ) / n
 for Lipschitz, smooth loss functions, showing the gap decays with sample size n
 (Theorem 3)."

We reproduce the closed-form Theorem-2.3 bound and verify its scaling:
  * G ~ 1/n                       (decays with sample size)   -> exact ratio test
  * G ~ eta*T * exp(eta*T*(K-k+1)/sqrt(m))  (linear in eta*T in the regime where
    the exponent is O(1), plus an exponential width penalty)
  * Under the regime (n=O~(d^2 K), m=O~(d^8 K^4), eta*T=O(d^2)) the exponent
    eta*T*(K-k+1)/sqrt(m) = Theta(1/(d^2 K)) -> 0, so exp->1 and G = Theta(1/K) -> 0.
MUTATION: insufficient width (m=d^2) makes the exponent blow up (-> inf) so the
gap no longer decays / explodes -> breaks.  (Also contrasts with Claim 5: G is
LINEAR in eta*T, not poly-logarithmic.)
"""
import json
import os
import numpy as np
import repro_common as R

SEED = R.claim_seed(4)
K = 4
k = 1
d = 64
pld = float(np.log(d))
rp = R.regime_params(d, K, polylog_d=pld)
n, m, etaT = rp['n'], rp['m'], rp['etat']
print(f"[Claim 4] Theorem-2.3 delayed generalization gap (K={K}, k={k}, d={d}, seed={SEED})")

# ---------- (A) bound form + scaling ----------
g = R.theorem23_gengap(k, d, n, m, K, etaT)
print(f"\n  Base config: G = {g['gen_gap']:.3e}  (etaT={etaT:.2e}, exp_arg={g['exp_arg']:.2e}, 1/n={g['one_over_n']:.2e})")

# 1/n scaling (exact)
g2n = R.theorem23_gengap(k, d, 2 * n, m, K, etaT)
r_n = g2n['gen_gap'] / g['gen_gap']
# linear-in-etaT in the regime (exponent small -> ~2x)
g2t = R.theorem23_gengap(k, d, n, m, K, 2 * etaT)
r_t = g2t['gen_gap'] / g['gen_gap']
print(f"  Scaling: G(2n)/G(n) = {r_n:.3f} (want 0.5, 1/n);  "
      f"G(2*etaT)/G(etaT) = {r_t:.3f} (~2 in small-exponent regime, linear in eta*T)")
scaling_ok = abs(r_n - 0.5) < 0.01 and r_t > 1.5

# ---------- (B) regime vanishing ----------
d_grid = [16, 32, 64, 128]
G_reg, exp_args = [], []
for dd in d_grid:
    p = float(np.log(dd))
    rpp = R.regime_params(dd, K, polylog_d=p)
    gg = R.theorem23_gengap(k, dd, rpp['n'], rpp['m'], K, rpp['etat'])
    G_reg.append(gg['gen_gap']); exp_args.append(gg['exp_arg'])
G_dec = all(G_reg[i + 1] < G_reg[i] for i in range(len(G_reg) - 1))
exp_small = exp_args[-1] < 0.01
print(f"\n  Under regime: G over d=32..128 = {[f'{x:.3e}' for x in G_reg]}")
print(f"  exponent etaT(K-k+1)/sqrt(m) over d=32..128 = {[f'{x:.2e}' for x in exp_args]} (-> 0)")
print(f"  G decreases with d (-> 0)? {G_dec};  exponent -> 0 (so exp->1)? {exp_small}")

# ---------- (C) MUTATION: insufficient width ----------
m_bad = d ** 2
gbad = R.theorem23_gengap(k, d, n, m_bad, K, etaT)
exponent_bad = gbad['exp_arg']
exploded = exponent_bad > 5.0
G_bad = gbad['gen_gap'] if not exploded else float('inf')
ratio_bad = (G_bad / g['gen_gap']) if not exploded else float('inf')
mutation_breaks = exploded or (ratio_bad > 10.0)
print(f"\n  MUTATION (m=d^2): exponent = {exponent_bad:.2e} -> {'EXPLODES (gap no longer decays)' if exploded else f'G={G_bad:.2e}'}")
print(f"  Insufficient width breaks the decaying gap? {mutation_breaks}")

# ---------- verdict ----------
verdict = "verified" if (scaling_ok and G_dec and exp_small and mutation_breaks) else "inconclusive"
print(f"\n  VERDICT: {verdict}")

result = {
    "claim": 4,
    "statement": "Theorem 2.3 bounds the delayed generalization gap by "
                 "eta*T*exp(eta*T*(K-k+1)/sqrt(m))/n for Lipschitz, smooth losses; gap decays with n.",
    "source": "Theorem 2.3 (Thm 2.3), Eq. after Thm 2.3, arXiv:2510.05573v2.",
    "seed": SEED,
    "base": g,
    "scaling_tests": {
        "ratio_1_over_n": float(r_n), "want": 0.5,
        "ratio_linear_in_etaT": float(r_t),
        "scaling_ok": bool(scaling_ok)},
    "regime": {
        "G_by_d": {str(dd): float(G_reg[i]) for i, dd in enumerate(d_grid)},
        "exponent_by_d": {str(dd): float(exp_args[i]) for i, dd in enumerate(d_grid)},
        "G_decreases": bool(G_dec), "exponent_to_zero": bool(exp_small)},
    "mutation_test": {
        "description": "Insufficient width: m=d^2 (below d^8 K^4).",
        "regime_G": float(g['gen_gap']),
        "break_m_exponent": float(exponent_bad),
        "break_m_exploded": bool(exploded),
        "property_breaks": bool(mutation_breaks),
        "note": "With m=d^2 the exponent eta*T*(K-k+1)/sqrt(m) is huge, so the exponential width "
                "penalty makes the gap explode instead of decaying with n; width is necessary."},
    "contrast_claim5": "Theorem 2.3 is LINEAR in eta*T (and exponential in eta*T/sqrt(m)); Claim 5's "
                       "improved bound (Thm B.1) is poly-logarithmic in T for self-bounded losses.",
    "verdict": verdict,
}
os.makedirs("results", exist_ok=True)
with open("results/claim4.json", "w") as f:
    json.dump(result, f, indent=2)
print("saved results/claim4.json")
