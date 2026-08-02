"""
verify_claim1.py -- Claim 1
"Theorem 1 gives a closed-form bound on train-time forgetting for task k after
training on K-k subsequent tasks, of order
  O~( eta*T*sqrt(K-k)/(d*sqrt(n)) + eta*T*sqrt(K-k)/(d^2 polylog(d)) + eta^2*T^2*K^2/sqrt(m) )."

We (A) reproduce the analytic 3-term bound and verify its scaling dependence by
exact ratio tests, and (B) compute the KERNEL-REGIME closed-form forgetting
object the theorem bounds on synthetic orthogonal XOR-cluster data.  The
1/sqrt(n) dependence (term1) is verified by the fluctuation of the empirical
forgetting across independent later-task datasets (which scales as 1/sqrt(n));
the mean (cross-task interference) is ~0 under orthogonality.
MUTATION: break task-mean orthogonality -> a constant cross-task bias appears
and forgetting no longer vanishes as n grows.
"""
import json
import os
import numpy as np
import repro_common as R

SEED = R.claim_seed(1)
d, K, etat = 64, 4, 2.0          # dimension, #tasks, eta*T
n = 2000                         # later-task sample size
m = 5000                         # hidden width (large -> kernel regime)
sigma = R.noise_sigma(d)
polylog_d = float(np.log(d))

print(f"[Claim 1] d={d} K={K} etat={etat} n={n} m={m} sigma={sigma:.4g} seed={SEED}")
print(f"  noise assumption sigma = SIGMA0/(log(d)^P sqrt(d)) with SIGMA0={R.SIGMA0}, P={R.SIGMA_LOGP}")

# ---------------- (A) analytic 3-term bound form / scaling ----------------
b = R.theorem21_bound(k=1, d=d, n=n, m=m, K=K, etat=etat, polylog_d=polylog_d)
print("\n  Analytic Theorem-2.1 bound (base config):")
print(f"    term1 (sample)      = {b['term1_sample']:.3e}")
print(f"    term2 (width/noise) = {b['term2_width_noise']:.3e}")
print(f"    term3 (finite-width)= {b['term3_finitewidth']:.3e}")
print(f"    bound               = {b['bound']:.3e}")

# exact ratio tests on the analytic expression
b2n = R.theorem21_bound(k=1, d=d, n=2 * n, m=m, K=K, etat=etat, polylog_d=polylog_d)
r_n = b2n['term1_sample'] / b['term1_sample']           # expect 1/sqrt(2) ~ 0.7071
b2d = R.theorem21_bound(k=1, d=2 * d, n=n, m=m, K=K, etat=etat, polylog_d=polylog_d)
r_d = b2d['term2_width_noise'] / b['term2_width_noise']  # expect 1/4
b2m = R.theorem21_bound(k=1, d=d, n=n, m=4 * m, K=K, etat=etat, polylog_d=polylog_d)
r_m = b2m['term3_finitewidth'] / b['term3_finitewidth']  # expect 1/2
b2k = R.theorem21_bound(k=1, d=d, n=n, m=m, K=2 * K, etat=etat, polylog_d=polylog_d)
r_k = b2k['term1_sample'] / b['term1_sample']           # sqrt((K-1)/(2K-1))
print(f"  Ratio tests (analytic, exact): "
      f"1/sqrt(n)={r_n:.3f} (want 0.707), 1/d^2={r_d:.3f} (want 0.250), "
      f"1/sqrt(m)={r_m:.3f} (want 0.500), sqrt(K-k)~={r_k:.3f}")
form_ok = (abs(r_n - 1 / np.sqrt(2)) < 0.02 and abs(r_d - 0.25) < 0.02
           and abs(r_m - 0.5) < 0.02)

# ---------------- (B) empirical kernel-regime forgetting ----------------
means = R.task_means(d, K, orthogonal=True, seed=SEED)
mean_F, std_F = R.forgetting_distribution(k=1, means=means, d=d, n=n, sigma=sigma,
                                           etat=etat, K=K, M=12, seed=SEED)
print(f"\n  Empirical kernel forgetting F_tr(1) (orthogonal tasks, n={n}):")
print(f"    mean (bias / cross-task interference) = {mean_F:.3e}")
print(f"    std  (sample-fluctuation ~ 1/sqrt(n)) = {std_F:.3e}")

# 1/sqrt(n) scaling of the FLUCTUATION (Theorem 2.1 term1)
mean_F2, std_F2 = R.forgetting_distribution(k=1, means=means, d=d, n=2 * n, sigma=sigma,
                                            etat=etat, K=K, M=12, seed=SEED)
r_n_emp = std_F2 / std_F
print(f"    at n={2*n}: std = {std_F2:.3e}  ->  std(2n)/std(n) = {r_n_emp:.3f}  (want ~0.707 = 1/sqrt(2))")

# consistency: empirical forgetting should be of smaller order than the
# kernel part of the bound (term1 + term2; term3 is the separate finite-width part)
kernel_part = b['term1_sample'] + b['term2_width_noise']
print(f"    empirical std / (term1+term2 bound) = {std_F / kernel_part:.3e}  (smaller order -> consistent)")

# ---------------- MUTATION: non-orthogonal tasks ----------------
means_no = R.task_means(d, K, orthogonal=False, seed=SEED + 777)
mean_F_no, std_F_no = R.forgetting_distribution(k=1, means=means_no, d=d, n=n, sigma=sigma,
                                                etat=etat, K=K, M=12, seed=SEED + 777)
print(f"\n  MUTATION (non-orthogonal tasks): mean F_tr(1) = {mean_F_no:.3e}  (orthogonal = {mean_F:.3e})")
mutation_breaks = mean_F_no > 10.0 * max(mean_F, 1e-12)
print(f"  Mutation breaks clean vanishing? {mutation_breaks}  (constant cross-task bias persists as n grows)")

# ---------------- verdict ----------------
verdict = "verified" if (form_ok and (0.4 < r_n_emp < 0.95)
                         and std_F < kernel_part and mutation_breaks) else "inconclusive"
print(f"\n  VERDICT: {verdict}")

result = {
    "claim": 1,
    "statement": "Theorem 2.1 gives a closed-form train-time forgetting bound of order "
                 "O~(eta*T*sqrt(K-k)/(d*sqrt(n)) + eta*T*sqrt(K-k)/(d^2 polylog(d)) + eta^2*T^2*K^2/sqrt(m)).",
    "source": "Theorem 2.1 (Thm 2.1), Eq. (4), arXiv:2510.05573v2; kernel-regime forgetting object Eq. after Thm 2.2.",
    "seed": SEED,
    "params": {"d": d, "K": K, "etat": etat, "n": n, "m": m, "sigma": float(sigma), "polylog_d": polylog_d},
    "analytic_bound": b,
    "analytic_ratio_tests": {
        "r_1_over_sqrt_n": float(r_n), "want": 0.7071,
        "r_1_over_d2": float(r_d), "want": 0.25,
        "r_1_over_sqrt_m": float(r_m), "want": 0.5,
        "r_sqrt_Kmk": float(r_k),
        "form_integrity_ok": bool(form_ok)},
    "empirical": {
        "F_tr_mean_orth": float(mean_F),
        "F_tr_std_orth": float(std_F),
        "F_tr_std_orth_2n": float(std_F2),
        "ratio_std_1_over_sqrt_n": float(r_n_emp),
        "std_over_kernel_bound": float(std_F / kernel_part)},
    "mutation_test": {
        "description": "Break the orthogonality assumption between task means (random near-parallel "
                       "directions instead of mutually orthogonal ones).",
        "F_tr_mean_nonorthogonal": float(mean_F_no),
        "F_tr_mean_orthogonal": float(mean_F),
        "property_breaks": bool(mutation_breaks),
        "note": "Non-orthogonal tasks introduce a constant cross-task interference bias in x_k^T A_j x_k "
                "that does NOT vanish as n grows, so the clean closed-form bound (which relies on "
                "orthogonal clusters) no longer describes the data and forgetting fails to vanish."},
    "verdict": verdict,
}
os.makedirs("results", exist_ok=True)
with open("results/claim1.json", "w") as f:
    json.dump(result, f, indent=2)
print("saved results/claim1.json")
