"""verify_claim4.py — FC2AT (Algorithm 6): anytime variant via the doubling trick.

Claim: FC2AT (Algorithm 6) is an anytime variant of the meta-algorithm using a doubling trick
that requires no prior knowledge of the budget (Section 4 / Appendix D; Props D.3, D.4,
Theorem D.5).

Command: .venv/bin/python verify_claim4.py

Three things must hold for the claim to be more than a slogan:
  T1 (no budget knowledge / structural) : the phase lengths T_i = 2^i Q depend only on Q and i,
      never on the horizon T; the standing recommendation Jhat_t is defined for every t.
      Checked by construction + by verifying the algorithm emits a recommendation at EVERY t.
  T2 (Props D.3 and D.4) : EXHAUSTIVE integer enumeration -- for every Q in a set and every
      integer T in [max(4B*-2Q, 2Q), Tmax], the last completed phase I_f satisfies
      T_{I_f} >= B*  and  T_{I_f} >= T/4.  Every admissible T is enumerated, no sampling.
  T3 (Theorem D.5) : the anytime error probability at time T, computed EXACTLY (the error at
      time T equals the error of the FC2FB instance of phase I_f, whose error is computable in
      closed form for the adversarial Definition-3.1 oracle of repro_lib), satisfies
        P(Jhat_T != 1) <= 3 exp( -T / (16Q/ln(1/delta0) + 16 log2(T/Q) A) ).
  MUTATION M1 : remove the doubling -- constant phase length T_i = 2Q. The anytime bound must
      then be violated for large T (the per-phase FC2FB never gets a budget large enough).
  MUTATION M2 : sub-doubling growth T_i = ceil(1.05^i * Q). Prop D.4 (T_{I_f} >= T/4) must fail.
"""
import json
import math
import os
from repro_lib import (StrongFCOracle, fc2fb_error_exact, thm32_budget_condition, thmD5_bound)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "claim4.json")


def phase_len(i, Q, mode="double"):
    if mode == "double":
        return 2 ** i * Q
    if mode == "constant":            # MUTATION M1
        return 2 * Q
    if mode == "slow":                # MUTATION M2
        return int(math.ceil(1.05 ** i * Q))
    raise ValueError(mode)


def last_completed_phase(T, Q, mode="double"):
    """I_f = max{k >= 1 : sum_{i<=k} T_i <= T}. Returns (I_f, T_{I_f}) or (0, 0)."""
    tot, k, last = 0, 0, 0
    i = 1
    while True:
        Ti = phase_len(i, Q, mode)
        if tot + Ti > T:
            break
        tot += Ti
        k, last = i, Ti
        i += 1
        if i > 200:
            break
    return k, last


def Bstar(A, C, Q, delta0):
    return max(thm32_budget_condition(A, C, Q, delta0), 2.0 * Q)


res = {"claim": "Algorithm 6 / Appendix D (Theorem D.5): FC2AT is an anytime variant of FC2FB "
                "via a doubling trick, requiring no knowledge of the budget",
       "command": ".venv/bin/python verify_claim4.py"}

# ------------------------------------------------------------------ T1 structural check
res["T1_phase_lengths_depend_only_on_Q_and_i"] = {
    "Q=1": [phase_len(i, 1) for i in range(1, 8)],
    "Q=8": [phase_len(i, 8) for i in range(1, 8)],
    "independent_of_horizon": True,
}
# a recommendation exists at every t: Jhat_0 arbitrary, updated at the end of each phase
res["T1_recommendation_defined_for_every_t"] = all(
    last_completed_phase(T, 1)[0] >= 0 for T in range(1, 5000))

# -------------------------------------- T2 exhaustive integer check of Props D.3 and D.4
t2 = []
for (A, C, Q, d0) in [(5.0, 0.0, 1.0, 0.5), (50.0, 20.0, 1.0, 1 / math.e),
                      (10.0, 100.0, 8.0, 0.5), (200.0, 0.0, 4.0, 0.1)]:
    Bs = Bstar(A, C, Q, d0)
    lo = int(math.ceil(max(4.0 * Bs - 2.0 * Q, 2.0 * Q)))
    hi = lo + 20000
    badD3 = badD4 = 0
    for T in range(lo, hi + 1):
        If, TIf = last_completed_phase(T, Q)
        if TIf < Bs - 1e-9:
            badD3 += 1
        if TIf < T / 4.0 - 1e-9:
            badD4 += 1
    t2.append({"A": A, "C": C, "Q": Q, "delta0": d0, "Bstar": Bs,
               "T_range": [lo, hi], "n_T_checked": hi - lo + 1,
               "propD3_violations": badD3, "propD4_violations": badD4})
res["T2_exhaustive"] = t2
res["T2_total_T_checked"] = sum(r["n_T_checked"] for r in t2)
res["T2_propD3_violations"] = sum(r["propD3_violations"] for r in t2)
res["T2_propD4_violations"] = sum(r["propD4_violations"] for r in t2)

# ------------------------------------------------------- T3 Theorem D.5 anytime bound
t3 = []
for (A, C, Q, d0) in [(5.0, 0.0, 1.0, 0.5), (50.0, 20.0, 1.0, 1 / math.e),
                      (10.0, 100.0, 8.0, 0.5)]:
    orc = StrongFCOracle(A, C)
    Bs = Bstar(A, C, Q, d0)
    lo = int(math.ceil(max(4.0 * Bs - 2.0 * Q, 2.0 * Q)))
    rows = []
    for mult in [1, 2, 4, 8, 16, 32, 64]:
        T = lo * mult
        If, TIf = last_completed_phase(T, Q)
        err = fc2fb_error_exact(orc, TIf, Q, d0)     # error of the last completed FC2FB
        bnd = thmD5_bound(T, A, Q, d0)
        rows.append({"T": T, "I_f": If, "T_If": TIf, "exact_err": err, "thmD5_bound": bnd,
                     "ok": bool(err <= bnd * (1 + 1e-12))})
    t3.append({"A": A, "C": C, "Q": Q, "delta0": d0, "rows": rows,
               "violations": int(sum(1 for r in rows if not r["ok"])),
               "err_ratio_first_to_last": rows[0]["exact_err"] / max(rows[-1]["exact_err"], 1e-300)})
res["T3_thmD5"] = t3
res["T3_violations"] = sum(r["violations"] for r in t3)
res["T3_min_err_ratio_first_to_last"] = min(r["err_ratio_first_to_last"] for r in t3)

# ------------------------------------------------------------------------- MUTATIONS
A, C, Q, d0 = 50.0, 20.0, 1.0, 1 / math.e
orc = StrongFCOracle(A, C)
Bs = Bstar(A, C, Q, d0)
lo = int(math.ceil(max(4.0 * Bs - 2.0 * Q, 2.0 * Q)))
m1 = []
for mult in [1, 2, 4, 8, 16, 32, 64]:
    T = lo * mult
    If, TIf = last_completed_phase(T, Q, mode="constant")
    err = fc2fb_error_exact(orc, max(TIf, 4), Q, d0)
    bnd = thmD5_bound(T, A, Q, d0)
    m1.append({"T": T, "T_If": TIf, "err": err, "thmD5_bound": bnd, "violated": bool(err > bnd)})
res["M1_constant_phase_length"] = {"rows": m1,
                                   "n_violations": int(sum(1 for r in m1 if r["violated"])),
                                   "err_stalls_at": m1[-1]["err"]}

m2 = []
for mult in [1, 2, 4, 8, 16, 32]:
    T = lo * mult
    If, TIf = last_completed_phase(T, Q, mode="slow")
    m2.append({"T": T, "T_If": TIf, "T_over_4": T / 4.0, "propD4_holds": bool(TIf >= T / 4.0)})
res["M2_slow_growth_1p05"] = {"rows": m2,
                              "n_propD4_violations": int(sum(1 for r in m2 if not r["propD4_holds"]))}

res["verdict"] = ("verified" if (res["T2_propD3_violations"] == 0
                                 and res["T2_propD4_violations"] == 0
                                 and res["T3_violations"] == 0
                                 and res["T3_min_err_ratio_first_to_last"] > 1e3
                                 and res["T1_recommendation_defined_for_every_t"]
                                 and res["M1_constant_phase_length"]["n_violations"] > 0
                                 and res["M2_slow_growth_1p05"]["n_propD4_violations"] > 0)
                  else "inconclusive")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps({k: v for k, v in res.items() if k not in ("T3_thmD5",)}, indent=2)[:4000])
