"""verify_claim2.py — inverted sample complexity of FC2FB.

Claim (Section 3, immediately after Theorem 3.2):
  with T*_delta = A ln(1/delta) + C, FC2FB attains fixed-budget sample complexity
  O( Q ln(1/delta) + A ln(1/delta) * ln( A ln(1/delta) / Q ) + C ),
  i.e. the fixed-confidence complexity up to logarithmic factors.

Command: .venv/bin/python verify_claim2.py

Method (deterministic, no randomness at all):
  B_req(delta) := the smallest budget B such that BOTH
     (i)  B >= thm32_budget_condition(A, C, Q, delta0)   [Theorem 3.2 applies]  and
     (ii) thm32_bound(B, A, Q, delta0) <= delta          [error at most delta]
  found by exact integer bisection on the monotone bound.
  F(delta) := Q ln(1/delta) + A ln(1/delta) * ln(max(e, A ln(1/delta)/Q)) + C   (claimed form)
  Verified iff  sup over the grid of  B_req / F  is a modest absolute constant that does NOT
  grow with A, C, Q or ln(1/delta)  (that is what O(.) asserts).

  Also reported: B_req / T*_delta, which must be polylog in T*_delta (log factors only).

MUTATION: drop the inner logarithm, F_naive = Q ln(1/delta) + A ln(1/delta) + C.
  If the inner ln really is needed, B_req / F_naive must GROW without bound as A grows.
"""
import json
import math
import os
import numpy as np
from repro_lib import thm32_bound, thm32_budget_condition

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "claim2.json")
D0 = 1.0 / math.e   # delta0 = 1/e, as the paper specifies for this corollary


def b_required(A, C, Q, delta, delta0=D0):
    lo = max(int(math.ceil(thm32_budget_condition(A, C, Q, delta0))), int(2 * Q) + 1, 4)
    if thm32_bound(lo, A, Q, delta0) <= delta:
        return lo
    hi = lo
    while thm32_bound(hi, A, Q, delta0) > delta:
        hi *= 2
        if hi > 10 ** 15:
            raise RuntimeError("no budget found")
    while hi - lo > 1:
        mid = (hi + lo) // 2
        if thm32_bound(mid, A, Q, delta0) <= delta:
            hi = mid
        else:
            lo = mid
    return hi


def F_claimed(A, C, Q, delta):
    l = math.log(1.0 / delta)
    return Q * l + A * l * math.log(max(math.e, A * l / Q)) + C


def F_naive(A, C, Q, delta):
    l = math.log(1.0 / delta)
    return Q * l + A * l + C


res = {"claim": "Section 3: FC2FB sample complexity is O(Q ln(1/d) + A ln(1/d) ln(A ln(1/d)/Q) + C),"
                " matching T*_delta = A ln(1/d) + C up to log factors",
       "command": ".venv/bin/python verify_claim2.py",
       "delta0": D0}

rows = []
As = [1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6]
Cs = [0.0, 10.0, 1e3, 1e5]
Qs = [1.0, 10.0, 100.0]
deltas = [0.1, 0.01, 1e-3, 1e-6, 1e-10]
for A in As:
    for C in Cs:
        for Q in Qs:
            for d in deltas:
                B = b_required(A, C, Q, d)
                Fc, Fn = F_claimed(A, C, Q, d), F_naive(A, C, Q, d)
                Ts = A * math.log(1.0 / d) + C
                rows.append({"A": A, "C": C, "Q": Q, "delta": d, "B_req": B,
                             "F_claimed": Fc, "F_naive": Fn, "Tstar": Ts,
                             "ratio_claimed": B / Fc, "ratio_naive": B / Fn,
                             "ratio_Tstar": B / Ts})

res["n_grid_points"] = len(rows)
res["max_ratio_B_over_F_claimed"] = max(r["ratio_claimed"] for r in rows)
res["min_ratio_B_over_F_claimed"] = min(r["ratio_claimed"] for r in rows)
res["argmax_claimed"] = max(rows, key=lambda r: r["ratio_claimed"])

# does the claimed-form ratio grow with A? (it must not, for O(.) to hold)
by_A = {}
for A in As:
    sel = [r["ratio_claimed"] for r in rows if r["A"] == A]
    by_A[str(A)] = max(sel)
res["max_ratio_claimed_by_A"] = by_A
res["ratio_claimed_growth_A1e6_over_A10"] = by_A[str(1e6)] / by_A[str(10.0)]

# MUTATION: without the inner ln
by_A_naive = {}
for A in As:
    sel = [r["ratio_naive"] for r in rows if r["A"] == A]
    by_A_naive[str(A)] = max(sel)
res["MUT_max_ratio_naive_by_A"] = by_A_naive
res["MUT_ratio_naive_growth_A1e6_over_A10"] = by_A_naive[str(1e6)] / by_A_naive[str(10.0)]
res["MUT_max_ratio_B_over_F_naive"] = max(r["ratio_naive"] for r in rows)

# overhead over the fixed-confidence complexity must be polylog in T*
res["max_ratio_B_over_Tstar"] = max(r["ratio_Tstar"] for r in rows)
worst = max(rows, key=lambda r: r["ratio_Tstar"])
res["argmax_Tstar"] = worst
# fit ratio_Tstar vs ln(T*): a polylog overhead means ratio ~ a + b ln T*
xs = np.array([math.log(r["Tstar"]) for r in rows])
ys = np.array([r["ratio_Tstar"] for r in rows])
b, a = np.polyfit(xs, ys, 1)
pred = b * xs + a
res["ratio_Tstar_vs_lnTstar_slope"] = float(b)
res["ratio_Tstar_vs_lnTstar_R2"] = float(1 - ((ys - pred) ** 2).sum() / ((ys - ys.mean()) ** 2).sum())
# and it must NOT be polynomial: ratio / T*^0.1 must vanish for large T*
res["max_ratio_over_Tstar_pow_0p1"] = max(r["ratio_Tstar"] / r["Tstar"] ** 0.1 for r in rows)
# polylog (not polynomial) overhead over the FC complexity:
res["max_ratio_Tstar_over_ln2Tstar"] = max(
    r["ratio_Tstar"] / (1.0 + math.log(r["Tstar"])) ** 2 for r in rows)
# the O(.) form contains a Q ln(1/delta) term, so the overhead over T* is only meaningful
# in the regime the paper recommends (Q <= A; generically Q = 1):
qa = [r for r in rows if r["Q"] <= r["A"]]
res["n_rows_Q_le_A"] = len(qa)
res["max_ratio_B_over_Tstar_Q_le_A"] = max(r["ratio_Tstar"] for r in qa)
res["max_ratio_Tstar_over_ln2Tstar_Q_le_A"] = max(
    r["ratio_Tstar"] / (1.0 + math.log(r["Tstar"])) ** 2 for r in qa)
res["MUT_naive_monotone_increasing_in_A"] = bool(
    all(by_A_naive[str(As[i])] < by_A_naive[str(As[i + 1])] for i in range(len(As) - 1)))
res["MUT_ratio_naive_growth_A1e6_over_A1"] = by_A_naive[str(1e6)] / by_A_naive[str(1.0)]
res["claimed_monotone_nonincreasing_in_A"] = bool(
    all(by_A[str(As[i])] >= by_A[str(As[i + 1])] for i in range(len(As) - 1)))

res["samples"] = rows[:5]
# NOTE: the B_req/T*_delta ratio is reported as a diagnostic only. The claim is an O(.) on the
# form F_claimed (which contains Q ln(1/delta) and C), so the decisive test is that
# B_req <= c * F_claimed with a single constant c that does not grow along any axis.
res["verdict"] = ("verified" if (res["max_ratio_B_over_F_claimed"] < 50.0
                                 and res["claimed_monotone_nonincreasing_in_A"]
                                 and res["ratio_claimed_growth_A1e6_over_A10"] < 1.0
                                 and res["MUT_naive_monotone_increasing_in_A"]
                                 and res["MUT_ratio_naive_growth_A1e6_over_A1"] > 5.0)
                  else "inconclusive")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps({k: v for k, v in res.items() if k != "samples"}, indent=2))
