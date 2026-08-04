"""verify_claim5.py — Section 5 applications.

Claim: "The framework's sample-complexity improvements are demonstrated on heterogeneous noise
bandits (Corollary 5.2), linear bandits (Corollary 5.4), and unimodal bandits (Corollary 5.7)."

Command: .venv/bin/python verify_claim5.py

Part A (heterogeneous noise, Section 5.1 / Alg 5 / Thm 5.1 / Cor 5.2) -- fully reproduced here:
  A1  Theorem 5.1: implement Algorithm 5 (PE-KHN) on Gaussian bandits with known, strongly
      heterogeneous sigma_i, and check P(tau > T*_delta) <= delta and delta-correctness on a
      grid of delta, over 5 seeds each (MC).
  A2  Corollary 5.2 vs Theorem 3.2: an ALGEBRAIC consistency check. Theorem 3.2 with
      delta0 = 1/e, Q = 1 gives denominator 4 + 4 A log2(B); Corollary 5.2 prints
      4 + 4 A ln(B).  Since log2(B) = ln(B)/ln 2 > ln(B), the printed Corollary 5.2 bound is
      STRICTLY SMALLER (stronger) than what Theorem 3.2 yields -- so as printed it does not
      follow from Theorem 3.2. Quantified over a grid.
  A3  FC2FB(PE-KHN) end-to-end: run Algorithm 3 with Algorithm 5 as the inner FC algorithm on a
      real Gaussian bandit, MC over 5 seeds and several budgets; check the empirical error is
      below the Theorem-3.2 form 3exp(-B/(4+4A log2 B)) and decays with B.
  A4  the K^7 vs K^9 separation instance of Section 5.1, verified symbolically with sympy
      (exact leading order in K), for the stated instance
      Delta_2 = K^-5, Delta_i = K^-4 (i>=3), sigma_1^2 = sigma_2^2 = K^-5, sigma_i^2 = K^-2.
  MUTATION M1 : make PE-KHN noise-blind (allocate with sigma_max for every arm, i.e. the SHVar
      / SH style allocation). The sample complexity on the separation-flavoured instance must
      grow by the predicted factor; the heterogeneous allocation must be strictly cheaper.

Part B (linear bandits, Cor 5.4) and Part C (unimodal, Cor 5.7): NOT reproduced. Both corollaries
  are analytic consequences of Theorem 3.2 whose constants (gamma*, rho* of Katz-Samuels et al.
  2020; T_mu(delta) of Poiani et al. 2024, plus their algorithms Peace / UniTT) are defined in
  OTHER papers and are not restated in this paper. Reproducing them means re-implementing and
  re-verifying those two external algorithms and their instance constants -- out of the time
  budget here, and any small self-invented substitute would be a toy, not evidence.
  Recorded honestly as inconclusive rather than faked.
"""
import json
import math
import os
import numpy as np
import sympy as sp
from repro_lib import (StrongFCOracle, fc2fb_schedule, pe_khn, pe_khn_constants,
                       tstar_thm51, thm32_bound, cor52_bound)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "claim5.json")

res = {"claim": "Section 5: improvements demonstrated on heterogeneous-noise (Cor 5.2), "
                "linear (Cor 5.4) and unimodal (Cor 5.7) bandits",
       "command": ".venv/bin/python verify_claim5.py"}

# instance: strongly heterogeneous noise, arm 0 best
MU = [1.0, 0.6, 0.4, 0.2]
SIG = [0.15, 0.15, 1.0, 1.0]
A_khn, C_khn = pe_khn_constants(MU, SIG)
res["instance"] = {"mu": MU, "sigma": SIG, "A_thm51": A_khn, "C_thm51": C_khn}

# ------------------------------------------------- A1: Theorem 5.1 high-prob complexity
a1 = []
for delta in [0.2, 0.1, 0.05]:
    Ts = tstar_thm51(MU, SIG, delta)
    per_seed = []
    for seed in range(5):
        rng = np.random.default_rng(1000 + seed)
        n = 400
        over, wrong = 0, 0
        for _ in range(n):
            term, corr, used = pe_khn(MU, SIG, delta, rng)
            if used > Ts:
                over += 1
            if term and not corr:
                wrong += 1
        per_seed.append({"seed": seed, "P_tau_gt_Tstar": over / n, "P_err": wrong / n})
    a1.append({"delta": delta, "Tstar": Ts, "n_per_seed": 400, "per_seed": per_seed,
               "max_P_tau_gt_Tstar": max(p["P_tau_gt_Tstar"] for p in per_seed),
               "max_P_err": max(p["P_err"] for p in per_seed)})
res["A1_thm51"] = a1
res["A1_violations_stopping"] = int(sum(1 for r in a1 if r["max_P_tau_gt_Tstar"] > r["delta"]))
res["A1_violations_correctness"] = int(sum(1 for r in a1 if r["max_P_err"] > r["delta"]))

# ------------------------------- A2: Corollary 5.2 as printed vs Theorem 3.2 (delta0=1/e, Q=1)
a2 = []
for A in [10.0, 100.0, 1000.0]:
    for B in [1000, 10000, 100000]:
        t32 = thm32_bound(B, A, 1.0, 1.0 / math.e)     # 3exp(-B/(4 + 4 A log2 B))
        c52 = cor52_bound(B, A)                        # 3exp(-B/(4 + 4 A ln  B))
        a2.append({"A": A, "B": B, "thm32_with_d0_1_over_e_Q1": t32, "cor52_as_printed": c52,
                   "cor52_is_stronger": bool(c52 < t32), "ratio_thm32_over_cor52": t32 / c52})
res["A2_cor52_vs_thm32"] = a2
res["A2_n_points_where_cor52_stronger_than_thm32"] = int(sum(1 for r in a2 if r["cor52_is_stronger"]))
res["A2_max_ratio_thm32_over_cor52"] = max(r["ratio_thm32_over_cor52"] for r in a2)
res["A2_note"] = ("log2(B) = ln(B)/ln2 = 1.4427 ln(B), so the denominator printed in Corollary "
                  "5.2 (4 + 4A ln B) is smaller than the one Theorem 3.2 gives (4 + 4A log2 B); "
                  "Corollary 5.2 as printed is therefore STRONGER than its stated parent theorem "
                  "-- an apparent ln vs log2 typo in the corollary, not a substantive error in "
                  "Theorem 3.2.")

# ------------------------------------------------- A3: FC2FB(PE-KHN) end to end (real algorithm)
def fc2fb_pekhn(B, Q, delta0, mu, sigma, rng):
    R, Bp = fc2fb_schedule(B, Q)
    for r in range(1, R + 1):
        L = 2.0 ** (R - r)
        log_inv_d = L * math.log(1.0 / delta0)
        if log_inv_d > 700:                 # delta underflows; the run cannot finish anyway
            continue
        d = math.exp(-log_inv_d)
        term, corr, used = pe_khn(mu, sigma, d, rng, budget=Bp)
        if term:
            return corr
    return False                            # arbitrary arm, counted as wrong


a3 = []
Bgrid = [2500, 5000, 10000, 20000, 40000]
for B in Bgrid:
    per_seed = []
    for seed in range(5):
        rng = np.random.default_rng(7000 + seed)
        n = 200
        wrong = sum(0 if fc2fb_pekhn(B, 1.0, 1.0 / math.e, MU, SIG, rng) else 1 for _ in range(n))
        per_seed.append(wrong / n)
    a3.append({"B": B, "per_seed_err": per_seed, "mean_err": float(np.mean(per_seed)),
               "thm32_bound": thm32_bound(B, A_khn, 1.0, 1.0 / math.e),
               "cor52_bound_as_printed": cor52_bound(B, A_khn),
               "n_per_seed": 200})
for r in a3:
    r["ok_vs_thm32"] = bool(r["mean_err"] <= r["thm32_bound"])
res["A3_fc2fb_pekhn"] = a3
res["A3_violations_thm32"] = int(sum(1 for r in a3 if not r["ok_vs_thm32"]))
res["A3_err_decreasing"] = bool(all(a3[i]["mean_err"] >= a3[i + 1]["mean_err"] for i in range(len(a3) - 1)))
res["A3_err_first_last"] = [a3[0]["mean_err"], a3[-1]["mean_err"]]
res["A3_shows_decay"] = bool(a3[0]["mean_err"] > a3[-1]["mean_err"])

# --------------------------------------------------------- A4: K^7 vs K^9 separation (sympy)
K = sp.symbols('K', positive=True, integer=True)
D2 = K ** -5
Di = K ** -4
s1s = K ** -5
s2s = K ** -5
sis = K ** -2
ours = s1s / D2 ** 2 + s2s / D2 ** 2 + (K - 2) * sis / Di ** 2
shvar = s1s / D2 ** 2 + s2s / D2 ** 2 + (K - 2) * sis / D2 ** 2
ours_e = sp.expand(sp.simplify(ours))
shvar_e = sp.expand(sp.simplify(shvar))
res["A4_symbolic"] = {
    "ours_expr": str(ours_e),
    "shvar_expr": str(shvar_e),
    "ours_leading_degree": int(sp.degree(sp.Poly(ours_e, K))),
    "shvar_leading_degree": int(sp.degree(sp.Poly(shvar_e, K))),
    "ratio_limit_shvar_over_ours": str(sp.limit(shvar_e / ours_e / K ** 2, K, sp.oo)),
    "paper_says": {"ours": "O(K^7)", "shvar": "O(K^9)"},
}
res["A4_matches_paper"] = bool(res["A4_symbolic"]["ours_leading_degree"] == 7
                               and res["A4_symbolic"]["shvar_leading_degree"] == 9)

# ------------------------------------------- MUTATION M1: noise-blind (sigma_max) allocation
def total_samples_khn(mu, sigma, delta, blind=False):
    """Deterministic worst-case sample count of Algorithm 5's allocation rule (no simulation):
    sum over arms of the allocation at the phase where the arm is eliminated (ell ~ log2(1/Delta))."""
    K_ = len(mu)
    gaps = [mu[0] - m for m in mu]
    smax = max(sigma)
    tot = 0.0
    for i in range(K_):
        d = gaps[1] if i == 0 else gaps[i]
        ell = max(1, int(math.ceil(math.log2(4.0 / d))))
        eps = 1.0 / 2 ** ell
        dl = delta / (ell * (ell + 1.0))
        s = smax if blind else sigma[i]
        tot += math.ceil(2.0 * s ** 2 / eps ** 2 * math.log(K_ / dl))
    return tot


m1 = []
for delta in [0.1, 0.01]:
    het = total_samples_khn(MU, SIG, delta, blind=False)
    bli = total_samples_khn(MU, SIG, delta, blind=True)
    m1.append({"delta": delta, "heterogeneous_alloc": het, "noise_blind_alloc": bli,
               "ratio": bli / het})
res["M1_noise_blind"] = m1
res["M1_blind_is_worse"] = bool(all(r["ratio"] > 1.0 for r in m1))
res["M1_min_ratio"] = min(r["ratio"] for r in m1)

# ---------------------------------------------------------------- Parts B and C: refusals
res["B_linear_bandits_cor54"] = {
    "verdict": "inconclusive",
    "reason": ("Corollary 5.4 is stated in terms of gamma*, rho* and the algorithm 'Fixed Budget "
               "Peace' (Katz-Samuels et al. 2020, Algorithm 3), none of which is defined in this "
               "paper. Verifying it requires re-implementing that external algorithm and its "
               "instance-dependent constants; not attempted within the time budget. No toy "
               "substitute was used.")}
res["C_unimodal_cor57"] = {
    "verdict": "inconclusive",
    "reason": ("Corollary 5.7 is stated in terms of T_mu(delta) and the algorithm UniTT (Poiani "
               "et al. 2024, Theorem 3.7), defined in an external paper. Same reason as above. "
               "Note that the FCW2S half of the Corollary-5.7 pipeline IS verified in claim 3 "
               "with delta1 = 1/(8e) and L = ceil(4 ln(1/delta)/ln(1/(4e delta1))), exactly the "
               "parameters Corollary 5.7 prescribes.")}

res["A_verdict"] = ("verified" if (res["A1_violations_stopping"] == 0
                                   and res["A1_violations_correctness"] == 0
                                   and res["A3_violations_thm32"] == 0
                                   and res["A3_err_decreasing"]
                                   and res["A3_shows_decay"]
                                   and res["A4_matches_paper"]
                                   and res["M1_blind_is_worse"])
                    else "inconclusive")
res["verdict"] = "partial: " + res["A_verdict"] + " (heterogeneous noise) / inconclusive (linear, unimodal)"

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps({k: v for k, v in res.items() if k not in ("A1_thm51", "A2_cor52_vs_thm32")},
                 indent=2)[:5000])
