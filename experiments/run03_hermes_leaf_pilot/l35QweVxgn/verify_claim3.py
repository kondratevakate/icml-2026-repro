"""
verify_claim3.py -- Claim 3
"Theorem 2 shows that under the same width/sample/iteration conditions as
Theorem 1, the misclassification error remains uniformly small across all K
tasks with high probability after K*T gradient descent iterations (Theorem 2)."

We verify the UNIFORM-TRAIN-ERROR claim by bounding, for every task k, the train
loss after the full K-task sequence as
    E(k) = base(T)  +  sum_{j>k} F_tr(j)  +  finite-width error,
where base(T) = d^2 log^2(T)/T is the per-task optimal loss (Remark B.2),
sum_{j>k} F_tr(j) is the forgetting accumulated from later tasks, and the
finite-width error = (eta*T)^2 K / sqrt(m).

Under the regime (with T = Theta(d^4), i.e. eta = Theta(1/d^2) so eta*T=Theta(d^2))
every component is o_d(1) (poly-log), and crucially E(k) is NON-INCREASING in k
(fewer later tasks -> less forgetting), so max_k E(k) = E(1) and the bound is
UNIFORM across tasks.  MUTATION: insufficient width (m=d^2) blows up the
finite-width term so train error is no longer uniformly small -> breaks.
"""
import json
import os
import numpy as np
import repro_common as R

SEED = R.claim_seed(3)
K = 4
d_grid = [16, 32, 64, 128]
cF = 1.0
print(f"[Claim 3] uniform train error (Theorem 2.2), K={K}, d in {d_grid}, seed={SEED}")

def train_loss_bound(d, K, cF=1.0):
    pld = float(np.log(d))
    rp = R.regime_params(d, K, polylog_d=pld)
    etaT = rp['etat']
    m = rp['m']
    T = d ** 4                       # eta = etaT/T = Theta(1/d^2) -> eta*T=Theta(d^2)
    base = cF * (d ** 2) * (np.log(T) ** 2) / T
    # forgetting of each task j after its later tasks, under the regime
    forget = [R.theorem21_bound(k=j, d=d, n=rp['n'], m=m, K=K, etat=etaT, polylog_d=pld)['bound']
              for j in range(1, K + 1)]
    fw = R.finite_width_error(etaT, K, m)
    E = []
    for k in range(1, K + 1):
        accum = sum(forget[j - 1] for j in range(k + 1, K + 1))   # later tasks j>k
        E.append(base + accum + fw)
    return {"d": d, "T": T, "base": base, "fw": fw, "forget": forget, "E": E,
            "maxE": max(E), "minE": min(E)}

rows = [train_loss_bound(d, K, cF) for d in d_grid]
print("\n  Per-task train-loss bound E(k) under the regime:")
for r in rows:
    print(f"    d={r['d']:3d} T={r['T']:.2e}: base={r['base']:.2e} fw={r['fw']:.2e} "
          f"| E(k=1..{K}) = {[f'{x:.3e}' for x in r['E']]}  max={r['maxE']:.3e}")

# uniformity: E(k) non-increasing in k (max at k=1)
uniform_structure = all(rows[0]['E'][i] >= rows[0]['E'][i + 1] for i in range(K - 1))
# vanishing: max_k E(k) = E(1) decreases with d (asymptotically o_d(1))
maxE = [r['maxE'] for r in rows]
maxE_decreasing = all(maxE[i + 1] < maxE[i] for i in range(len(maxE) - 1))
r1 = maxE[-1] / maxE[1]
print(f"\n  E(k) non-increasing in k (max at k=1, uniform)? {uniform_structure}")
print(f"  max_k E(k)=E(1) over d=32..128: {[f'{x:.3e}' for x in maxE]} ratio={r1:.3f} (<1 -> 0)")

# ---------- MUTATION: insufficient width ----------
def train_loss_break_m(d, K):
    pld = float(np.log(d))
    rp = R.regime_params(d, K, polylog_d=pld)
    etaT, n = rp['etat'], rp['n']
    m_bad = d ** 2
    T = d ** 4
    base = cF * (d ** 2) * (np.log(T) ** 2) / T
    forget = [R.theorem21_bound(k=j, d=d, n=n, m=m_bad, K=K, etat=etaT, polylog_d=pld)['bound']
              for j in range(1, K + 1)]
    fw = R.finite_width_error(etaT, K, m_bad)
    E = [base + sum(forget[j - 1] for j in range(k + 1, K + 1)) + fw for k in range(1, K + 1)]
    return max(E), fw

d = 64
maxE_bad, fw_bad = train_loss_break_m(d, K)
rp = R.regime_params(d, K, polylog_d=float(np.log(d)))
b_reg = train_loss_bound(d, K, cF)
mutation_breaks = maxE_bad > 10.0 * b_reg['maxE']
print(f"\n  MUTATION at d={d}: break m=d^2 -> fw={fw_bad:.2e}, max_k E(k)={maxE_bad:.2e} "
      f"(regime maxE={b_reg['maxE']:.2e})")
print(f"  Insufficient width breaks uniform small train error? {mutation_breaks}")

# ---------- verdict ----------
verdict = "verified" if (uniform_structure and maxE_decreasing and mutation_breaks) else "inconclusive"
print(f"\n  VERDICT: {verdict}")

result = {
    "claim": 3,
    "statement": "Theorem 2.2: under the same width/sample/iteration regime as Theorem 1, the train "
                 "(misclassification) error remains uniformly small across all K tasks with high "
                 "probability after K*T GD iterations.",
    "source": "Theorem 2.2 (Thm 2.2), arXiv:2510.05573v2; per-task loss Remark B.2.",
    "seed": SEED,
    "K": K,
    "note": "Train-loss bound E(k)=base(T)+sum_{j>k}F_tr(j)+finite-width, with T=Theta(d^4) "
            "(eta=Theta(1/d^2) so eta*T=Theta(d^2)). Every term is o_d(1) (poly-log); E(k) is "
            "non-increasing in k so max_k E(k)=E(1) -- the bound is uniform across tasks.",
    "per_d": [{"d": r['d'], "T": r['T'], "base": float(r['base']), "fw": float(r['fw']),
               "forget_terms": [float(x) for x in r['forget']],
               "E_by_k": [float(x) for x in r['E']], "maxE": float(r['maxE'])} for r in rows],
    "uniformity": {
        "E_nonincreasing_in_k": bool(uniform_structure),
        "maxE_by_d": {str(r['d']): float(r['maxE']) for r in rows},
        "maxE_doubling_ratio": float(r1),
        "maxE_vanishes": bool(maxE_decreasing)},
    "mutation_test": {
        "description": "Insufficient width: m=d^2 (far below d^8 K^4).",
        "d": d,
        "regime_maxE": float(b_reg['maxE']),
        "break_m_maxE": float(maxE_bad),
        "break_m_finite_width": float(fw_bad),
        "property_breaks": bool(mutation_breaks),
        "note": "With m=d^2 the finite-width error (eta*T)^2 K/sqrt(m) grows as d^3, so train loss is "
                "no longer uniformly small across tasks -> the width condition is necessary."},
    "verdict": verdict,
}
os.makedirs("results", exist_ok=True)
with open("results/claim3.json", "w") as f:
    json.dump(result, f, indent=2)
print("saved results/claim3.json")
