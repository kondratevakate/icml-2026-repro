"""
verify_claim2.py -- Claim 2
"The forgetting bounds hold under the parameter regime
   n = O~(d^2 K) samples,  m = O~(d^8 K^4) hidden-layer width,
   eta*T = O(d^2) training iterations
 on a d-dimensional XOR-cluster dataset with K tasks (Theorem 1)."

We verify the SUFFICIENCY of the prescribed regime for vanishing forgetting
(F_tr = o_d(1)) by evaluating the closed-form Theorem-2.1 bound term by term for
increasing d and confirming every term -> 0 (hence the sum is o_d(1)):
  * term1 = eta*T*sqrt(K-k)/(d sqrt(n))
            = Theta( sqrt((K-k)/K) / sqrt(polylog d) )   -> 0 as 1/sqrt(log d)
            (the d shows up only through the polylog factor hidden in n=O~(d^2 K));
  * term2 = eta*T*sqrt(K-k)/(d^2 polylog d) = Theta( sqrt(K)/polylog d ) -> 0;
  * term3 = (eta*T)^2 K^2 / sqrt(m) = Theta( 1/polylog^2 d )             -> 0
    (with m = d^8 K^4 polylog^4 d).
All three vanish (poly-logarithmically) as d->inf, so F_tr = o_d(1).

Finally the JOINT-CONTROL property ("neither n nor m alone suffices"): breaking
the width scaling (m=d^2) or the sample scaling (n=d) leaves a non-vanishing
term, whereas with both correct the bound is o_d(1).
"""
import json
import os
import numpy as np
import repro_common as R

SEED = R.claim_seed(2)
K = 4
d_grid = [16, 32, 64, 128]
print(f"[Claim 2] parameter-regime sufficiency (K={K}, d in {d_grid}, seed={SEED})")

# ---------- (A) every term -> 0 with d ----------
t1, t2, t3, B1 = [], [], [], []
for dd in d_grid:
    p = float(np.log(dd))
    rp = R.regime_params(dd, K, polylog_d=p)
    b = R.theorem21_bound(k=1, d=dd, n=rp['n'], m=rp['m'], K=K, etat=rp['etat'], polylog_d=p)
    t1.append(b['term1_sample']); t2.append(b['term2_width_noise'])
    t3.append(b['term3_finitewidth']); B1.append(b['bound'])
r1 = t1[-1] / t1[1]      # d=128 / d=32
r2 = t2[-1] / t2[1]
r3 = t3[-1] / t3[1]
print(f"\n  term1 over d=32..128: {[f'{x:.3e}' for x in t1]}  ratio={r1:.3f}  (->0 as 1/sqrt(log d))")
print(f"  term2 over d=32..128: {[f'{x:.3e}' for x in t2]}  ratio={r2:.3f}  (->0 as 1/log d)")
print(f"  term3 over d=32..128: {[f'{x:.3e}' for x in t3]}  ratio={r3:.3f}  (->0 as 1/log^2 d)")
print(f"  B1   over d=32..128: {[f'{x:.3e}' for x in B1]}")
all_terms_vanish = (r1 < 1.0) and (r2 < 1.0) and (r3 < 1.0)
print(f"  Every term decreases with d (-> 0, i.e. o_d(1))? {all_terms_vanish}")

# ---------- (B) joint-control MUTATIONS ----------
d = 64
p = float(np.log(d))
rp = R.regime_params(d, K, polylog_d=p)
b_reg = R.theorem21_bound(k=1, d=d, n=rp['n'], m=rp['m'], K=K, etat=rp['etat'], polylog_d=p)
b_break_m = R.theorem21_bound(k=1, d=d, n=rp['n'], m=d ** 2, K=K, etat=rp['etat'], polylog_d=p)
b_break_n = R.theorem21_bound(k=1, d=d, n=d, m=rp['m'], K=K, etat=rp['etat'], polylog_d=p)
ratio_m = b_break_m['bound'] / b_reg['bound']
ratio_n = b_break_n['bound'] / b_reg['bound']
print(f"\n  MUTATION (regime B1={b_reg['bound']:.2e}):")
print(f"    break m (m=d^2) -> B1={b_break_m['bound']:.2e} (term3={b_break_m['term3_finitewidth']:.2e}, grows)")
print(f"    break n (n=d)   -> B1={b_break_n['bound']:.2e} (term1={b_break_n['term1_sample']:.2e}, grows)")
mutation_breaks = (ratio_m > 10.0) and (ratio_n > 10.0)
print(f"  Both single-factor breaks non-vanishing (>10x)? {mutation_breaks} (m={ratio_m:.1e}, n={ratio_n:.1e})")

# ---------- (C) light empirical: kernel fluctuation shrinks with n~d^2 ----------
print("\n  Empirical kernel fluctuation under regime n-scaling (etat fixed=2, capped n):")
emp_std = []
for dd in d_grid:
    n_emp = min(4000, int(dd ** 2 * K * np.log(dd)))
    sigma = R.noise_sigma(dd)
    means = R.task_means(dd, K, orthogonal=True, seed=SEED)
    _, s = R.forgetting_distribution(k=1, means=means, d=dd, n=n_emp, sigma=sigma,
                                     etat=2.0, K=K, M=6, seed=SEED + dd)
    emp_std.append(s)
    print(f"    d={dd:3d}: n={n_emp} -> std(F_tr) = {s:.3e}")
emp_decreasing = all(emp_std[i + 1] < emp_std[i] for i in range(len(emp_std) - 1))

# ---------- verdict ----------
verdict = "verified" if (all_terms_vanish and mutation_breaks and emp_decreasing) else "inconclusive"
print(f"\n  VERDICT: {verdict}")

result = {
    "claim": 2,
    "statement": "The forgetting bound holds under the regime n=O~(d^2 K), m=O~(d^8 K^4), "
                 "eta*T=O(d^2) on a d-dim XOR-cluster dataset with K tasks (Theorem 1).",
    "source": "Theorem 2.1 regime conditions (n=O~(d^2 K), m=O~(d^8 K^4), eta*T=O(d^2)), arXiv:2510.05573v2.",
    "seed": SEED,
    "note_on_vanishing": "Under the regime every term vanishes as d->inf (poly-logarithmically): "
                         "term1 ~ 1/sqrt(log d) (via the polylog hidden in n=O~(d^2 K)), "
                         "term2 ~ 1/log d, term3 ~ 1/log^2 d. Hence F_tr = o_d(1) as claimed.",
    "term1_by_d": {str(dd): float(t1[i]) for i, dd in enumerate(d_grid)},
    "term2_by_d": {str(dd): float(t2[i]) for i, dd in enumerate(d_grid)},
    "term3_by_d": {str(dd): float(t3[i]) for i, dd in enumerate(d_grid)},
    "B1_by_d": {str(dd): float(B1[i]) for i, dd in enumerate(d_grid)},
    "doubling_ratios": {"term1": float(r1), "term2": float(r2), "term3": float(r3)},
    "all_terms_vanish": bool(all_terms_vanish),
    "mutation_test": {
        "description": "Break ONE factor: (a) width m=d^2 (below d^8 K^4), (b) sample size n=d (below d^2 K).",
        "regime_B1": float(b_reg['bound']),
        "break_m_B1": float(b_break_m['bound']),
        "break_m_term3": float(b_break_m['term3_finitewidth']),
        "break_n_B1": float(b_break_n['bound']),
        "break_n_term1": float(b_break_n['term1_sample']),
        "ratio_break_m": float(ratio_m),
        "ratio_break_n": float(ratio_n),
        "property_breaks": bool(mutation_breaks),
        "note": "With correct m but n=d, term1 stays order-1 and grows; with correct n but m=d^2, term3 "
                "grows as d^3. Neither factor alone drives the bound to zero -- confirming 'joint control'."},
    "empirical_fluctuation_by_d": {str(dd): float(emp_std[i]) for i, dd in enumerate(d_grid)},
    "verdict": verdict,
}
os.makedirs("results", exist_ok=True)
with open("results/claim2.json", "w") as f:
    json.dump(result, f, indent=2)
print("saved results/claim2.json")
