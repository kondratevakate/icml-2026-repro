"""verify_claim3.py — FCW2S (Algorithm 4) converts a weak FC algorithm into a strong one.

Claim: FCW2S (Algorithm 4) converts a weak fixed-confidence algorithm (Definition 4.1) into a
strong one (Definition 3.1) via parallel execution and majority voting (Section 4;
Propositions 4.2 and 4.3).

Command: .venv/bin/python verify_claim3.py

Weak FC oracle (saturates Definition 4.1 with equality, adversarially):
  each of the L independent instances, run with the fixed base rate delta0, independently
    * w.p. delta0     : terminates immediately (internal step 1) recommending arm 2 (WRONG)
                        -- and every wrong instance recommends the SAME wrong arm 2,
                           the worst case for majority voting
    * w.p. delta0     : never terminates
    * w.p. 1 - 2delta0: terminates at internal step f(delta0) with arm 1 (correct)
  so P(err, tau<inf) = delta0 and P(tau > f(delta0)) = delta0, exactly Definition 4.1.

Two independent computations:
  (a) EXACT: the outcome of Algorithm 4 depends on the randomness only through the triple
      (n_wrong, n_correct, n_never) with n_wrong + n_correct + n_never = L, so the error
      probability and the stopping-time tail are computed by EXHAUSTIVE enumeration over the
      whole (finite) multinomial support -- no seeds involved.
  (b) Monte Carlo: a step-by-step event simulation of Algorithm 4 (always sample the surviving
      instance with the smallest internal time step; stop when |S| < floor(L/2); majority vote),
      over 5 seeds, to confirm the exact model matches the literal pseudocode.

Checks:
  T1 Prop 4.2: P(Jhat != 1, tau < inf) <= (4 e delta0)^(L/4) and <= delta,
     for L = ceil(4 ln(1/delta)/ln(1/(4 e delta0))), over a delta grid.
  T2 Prop 4.3: P(tau > L f(delta0)) <= (2 e delta0)^(L/2) and <= delta.
  T3 Strongness (Definition 3.1): the resulting T*_delta = L f(delta0) is LINEAR in ln(1/delta)
     -- fit T*_delta on ln(1/delta), require R^2 ~ 1 and slope ~ 4 f(delta0)/ln(1/(4e delta0)).
  MUTATION M1: replace majority voting by "output the first instance that terminated".
     Prediction: error stops decaying with L and stalls near delta0 (weak, not strong).
  MUTATION M2: keep voting but let each wrong instance vote for a DIFFERENT wrong arm
     (i.e. votes split) -- error must become smaller than the tight worst case, showing that
     the worst case used in (a) really is the worst case (sanity, not a break).
"""
import json
import math
import os
from itertools import product
import numpy as np
from repro_lib import prop42_bound, prop43_bound

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "claim3.json")

DELTA0 = 1.0 / (8.0 * math.e)      # the value the paper itself uses in Corollary 5.7
F = 100                            # f(delta0), the weak algorithm's sample complexity


def L_of(delta, delta0=DELTA0):
    return int(math.ceil(4.0 * math.log(1.0 / delta) / math.log(1.0 / (4.0 * math.e * delta0))))


def exact_stats(L, delta0=DELTA0, mode="majority"):
    """Exhaustive enumeration over (n_wrong, n_correct, n_never)."""
    from math import lgamma
    p = [delta0, 1.0 - 2.0 * delta0, delta0]         # wrong, correct, never
    lg = lambda n: lgamma(n + 1)
    err = 0.0
    tail = 0.0                                        # P(tau > L f(delta0))
    for nw in range(L + 1):
        for nc in range(L - nw + 1):
            nn = L - nw - nc
            lp = lg(L) - lg(nw) - lg(nc) - lg(nn)
            for k, n in ((0, nw), (1, nc), (2, nn)):
                if n:
                    lp += n * math.log(p[k])
            prob = math.exp(lp)
            # stopping: instances terminate; algorithm stops as soon as |S| < floor(L/2).
            # wrong ones terminate at internal step 1, correct ones at step f(delta0),
            # never-terminating ones run forever. By time L*f(delta0) every wrong and
            # every correct instance has terminated, so |S| = nn.
            stopped_in_time = (nn < math.floor(L / 2.0))
            if not stopped_in_time:
                tail += prob
                # the algorithm has not stopped: no output yet -> counted as an error only
                # in the joint event of Prop 4.2 ({Jhat != 1, tau < inf}), so skip.
                continue
            if mode == "majority":
                wrong_wins = (nw >= nc)               # tie broken adversarially
            elif mode == "first":                     # MUTATION M1
                # the first instance to terminate is a wrong one whenever nw >= 1
                wrong_wins = (nw >= 1)
            elif mode == "split":                     # MUTATION M2
                wrong_wins = (nw > 0 and math.ceil(nw / max(1, min(nw, 3))) >= nc)
            else:
                raise ValueError(mode)
            if wrong_wins:
                err += prob
    return err, tail


def simulate(L, delta0=DELTA0, f=F, n=20000, seed=0, mode="majority"):
    """Literal event simulation of Algorithm 4."""
    rng = np.random.default_rng(seed)
    wrong_cnt, tail_cnt = 0, 0
    for _ in range(n):
        u = rng.random(L)
        typ = np.where(u < delta0, 0, np.where(u < 2 * delta0, 2, 1))   # 0 wrong,1 correct,2 never
        term_step = np.where(typ == 0, 1, np.where(typ == 1, f, np.inf))
        # round-robin on smallest internal time: global stopping time is the point at which
        # the (L - floor(L/2))-th instance terminates. Order terminations by internal step.
        order = np.sort(term_step)
        # |S| < floor(L/2)  <=>  #terminated > L - floor(L/2)
        need = L - int(math.floor(L / 2.0)) + 1
        if need > L or not np.isfinite(order[need - 1]):
            tail_cnt += 1
            continue
        thresh = order[need - 1]
        terminated = term_step <= thresh
        nw = int(((typ == 0) & terminated).sum())
        nc = int(((typ == 1) & terminated).sum())
        if mode == "first":
            win_wrong = nw >= 1
        else:
            win_wrong = nw >= nc
        if win_wrong:
            wrong_cnt += 1
    return wrong_cnt / n, tail_cnt / n


res = {"claim": "Section 4 / Algorithm 4 / Prop 4.2 + 4.3: FCW2S converts a weak FC algorithm "
                "(Def 4.1) into a strong FC algorithm (Def 3.1) by parallel runs + majority vote",
       "command": ".venv/bin/python verify_claim3.py",
       "delta0": DELTA0, "f_delta0": F}

# -------------------------------------------------- T1 / T2 over a grid of target deltas
rows = []
for delta in [0.5, 0.2, 0.1, 0.01, 1e-3, 1e-4, 1e-6, 1e-8]:
    L = L_of(delta)
    err, tail = exact_stats(L)
    rows.append({"delta": delta, "L": L, "exact_err": err, "prop42_bound": prop42_bound(L, DELTA0),
                 "exact_tail": tail, "prop43_bound": prop43_bound(L, DELTA0),
                 "Tstar": L * F})
res["grid"] = rows
res["T1_prop42_violations"] = int(sum(1 for r in rows if r["exact_err"] > r["prop42_bound"] * (1 + 1e-12)))
res["T1_prop42_vs_delta_violations"] = int(sum(1 for r in rows if r["exact_err"] > r["delta"] * (1 + 1e-12)))
res["T2_prop43_violations"] = int(sum(1 for r in rows if r["exact_tail"] > r["prop43_bound"] * (1 + 1e-12)))
res["T2_prop43_vs_delta_violations"] = int(sum(1 for r in rows if r["exact_tail"] > r["delta"] * (1 + 1e-12)))

# ------------------------------------------------------ T3 strongness: T* linear in ln(1/delta)
x = np.array([math.log(1.0 / r["delta"]) for r in rows])
y = np.array([float(r["Tstar"]) for r in rows])
sl, ic = np.polyfit(x, y, 1)
pred = sl * x + ic
res["T3_slope_A_empirical"] = float(sl)
res["T3_slope_A_predicted"] = 4.0 * F / math.log(1.0 / (4.0 * math.e * DELTA0))
res["T3_R2"] = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
res["T3_intercept_C"] = float(ic)

# --------------------------------------------------------------- Monte Carlo cross-check
mc = []
Lchk = L_of(1e-4)
for seed in range(5):
    e, t = simulate(Lchk, n=20000, seed=seed)
    mc.append({"seed": seed, "mc_err": e, "mc_tail": t})
ex_e, ex_t = exact_stats(Lchk)
res["MC_L"] = Lchk
res["MC_runs"] = mc
res["MC_mean_err"] = float(np.mean([m["mc_err"] for m in mc]))
res["MC_mean_tail"] = float(np.mean([m["mc_tail"] for m in mc]))
res["MC_exact_err"] = ex_e
res["MC_exact_tail"] = ex_t
res["MC_err_abs_dev"] = abs(res["MC_mean_err"] - ex_e)
res["MC_tail_abs_dev"] = abs(res["MC_mean_tail"] - ex_t)

# ------------------------------------------------------------------------- MUTATION M1
m1 = []
for delta in [0.1, 0.01, 1e-3, 1e-4, 1e-6, 1e-8]:
    L = L_of(delta)
    e, _ = exact_stats(L, mode="first")
    m1.append({"delta": delta, "L": L, "err_first_terminated": e, "prop42_bound": prop42_bound(L, DELTA0)})
res["M1_first_terminated_wins"] = m1
res["M1_violations_of_prop42"] = int(sum(1 for r in m1 if r["err_first_terminated"] > r["prop42_bound"]))
res["M1_error_at_largest_L"] = m1[-1]["err_first_terminated"]
res["M1_error_ratio_first_to_last"] = m1[0]["err_first_terminated"] / m1[-1]["err_first_terminated"]
res["M1_stalls_near_delta0"] = bool(m1[-1]["err_first_terminated"] > 0.5 * DELTA0)

res["verdict"] = ("verified" if (res["T1_prop42_violations"] == 0
                                 and res["T1_prop42_vs_delta_violations"] == 0
                                 and res["T2_prop43_violations"] == 0
                                 and res["T2_prop43_vs_delta_violations"] == 0
                                 and res["T3_R2"] > 0.999
                                 and res["MC_err_abs_dev"] < 5e-3
                                 and res["MC_tail_abs_dev"] < 5e-3
                                 and res["M1_violations_of_prop42"] > 0
                                 and res["M1_stalls_near_delta0"])
                  else "inconclusive")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2)[:5000])
