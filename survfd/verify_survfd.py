"""
Reproduction of the two claims of:
  "Functional Decomposition and Shapley Interactions for Interpreting Survival Models"
  Langbein, Baniecki, Fumagalli, Koenen, Wright, Herbinger. ICML 2026 (orid SldP4LGjdz),
  arXiv 2602.16505.

Claim 1: SurvFD separates higher-order effects into TIME-DEPENDENT and TIME-INDEPENDENT
         components for analyzing feature interactions in survival models.
Claim 2: SurvSHAP-IQ extends Shapley interactions to time-indexed functions, providing a
         practical estimator for higher-order, time-dependent interactions.

Formulas transcribed from the paper (p=2 features, brute-forceable, no invented decomposition):
  Hazard model (Eq. 3):        h(t|x) = h0(t) * exp(G(t|x))
  Ground-truth decomp (Eq. 8): log h(t|x) = b(t) + sum_{M in Id} gM(t|x) + sum_{M in Iid} gM(x)
  Pure effects, marginal FD (Eq. 4-5), F: X -> R at fixed t:
      f_empty = E_X[F(x)];  fM(x) = E_{X_Mbar}[F(x)] - sum_{Z subset M} fZ(x)
  Theorem 3.2: independent features + (i) G linear incl. interactions, or (ii) G purely additive
      main-effect model  =>  SurvFD recovers the ground-truth partition EXACTLY.
  Value function (Sec 3.3):    v(t|M) = E[S(t|X) | X_M = x_M] - E[S(t|X)]
  Shapley interaction index / n-Shapley (Eq. 10), discrete derivative:
      Delta_K(M) = sum_{L subset K} (-1)^(|K|-|L|) v(t | M union L)
      phi_K^(k)(t|x) = sum_{M subset P\\K} [1 / ((p-k+1) * C(p-k, |M|))] * Delta_K(M)

IMPORTANT experimental-design fix (v2): all Monte-Carlo reference-distribution expectations use
COMMON RANDOM NUMBERS (one fixed reference sample per run, reused across every t and every
subset). Without this, independently-redrawn MC samples at each t inject noise that is
indistinguishable from genuine time-dependence -- a first version of this script made exactly
that mistake and spuriously flagged every effect as "time-dependent". Threshold `tol` is set from
the actual Monte-Carlo standard error, not an arbitrary small number.

Design: p=2 independent features X1, X2 ~ U(-1,1), constant baseline hazard h0=0.1, four scenarios
spanning the paper's own taxonomy:
  A: additive, no interaction, time-independent        -> Iid={1,2}, Id={}   (Thm 3.2 case ii)
  B: additive, no interaction, one time-dependent main  -> Iid={1}, Id={2}   (Thm 3.2 case ii)
  C: linear WITH interaction, time-independent          -> Iid={1,2,{1,2}}  (Thm 3.2 case i)
  D: NONLINEAR main effect + interaction (boundary, outside Theorem 3.2) -> exact recovery not
     guaranteed; used to show "when and why additive explanations fail".
"""
import numpy as np
from itertools import combinations
from math import comb

rng = np.random.default_rng(0)
N_REF = 60000
TIMES = np.array([0.5, 2.0, 5.0])
H0 = 0.1
X_MC = rng.uniform(-1, 1, size=(N_REF, 2))          # ONE fixed reference sample for the whole run


def G_scenarios():
    return {
        "A_additive_TI_nointer":   (lambda t, x1, x2: 1.2 * x1 + 0.9 * x2,
                                    {"Iid": [(0,), (1,)], "Id": []}),
        "B_additive_TD_nointer":   (lambda t, x1, x2: 1.2 * x1 + 0.9 * x2 * np.log1p(t),
                                    {"Iid": [(0,)], "Id": [(1,)]}),
        "C_linear_TI_interaction": (lambda t, x1, x2: 1.2 * x1 + 0.9 * x2 + 0.8 * x1 * x2,
                                    {"Iid": [(0,), (1,), (0, 1)], "Id": []}),
        "D_nonlinear_boundary":    (lambda t, x1, x2: 1.2 * x1 + 0.9 * np.arctan(2 * x2) + 0.8 * x1 * x2,
                                    {"Iid": None, "Id": None}),
    }


def logh(G, t, x1, x2):
    return np.log(H0) + G(t, x1, x2)


def survival(G, t, x1, x2):
    return np.exp(-H0 * t * np.exp(G(t, x1, x2)))


# --------------------------------- SurvFD: pure effects (Eq. 4-5) --------------------------------
def pure_effects(F, t, x_ref, subsets):
    """Marginal-FD pure effects (Eq. 5) using the SHARED reference sample X_MC."""
    f = {(): float(np.mean(F(t, X_MC[:, 0], X_MC[:, 1])))}
    for M in sorted(subsets, key=len):
        if M == ():
            continue
        x1v = x_ref[0] if 0 in M else X_MC[:, 0]
        x2v = x_ref[1] if 1 in M else X_MC[:, 1]
        Ex = float(np.mean(F(t, x1v, x2v)))
        f[M] = Ex - sum(f[Z] for Z in f if set(Z) < set(M))
    return f


def mc_standard_error(F, t, subsets):
    """Estimate the MC noise floor for f_empty (dominant source of estimation error)."""
    vals = F(t, X_MC[:, 0], X_MC[:, 1])
    return float(np.std(vals) / np.sqrt(N_REF))


def classify(F, x_ref, subsets, times=TIMES):
    vals = {t: pure_effects(F, t, x_ref, subsets) for t in times}
    se = mc_standard_error(F, times[0], subsets)
    tol = max(6.0 * se, 1e-4)          # 6 MC standard errors -> < 1e-6 false-positive rate per test
    verdict = {}
    for M in subsets:
        series = np.array([vals[t][M] for t in times])
        verdict[M] = "TD" if (series.max() - series.min()) > tol else "TI"
    return verdict, vals, tol


# ------------------------------- SurvSHAP-IQ: n-Shapley (Eq. 10) --------------------------------
def value_fn(F_S, t, x_ref, M):
    x1v = x_ref[0] if 0 in M else X_MC[:, 0]
    x2v = x_ref[1] if 1 in M else X_MC[:, 1]
    cond = float(np.mean(F_S(t, x1v, x2v)))
    marg = float(np.mean(F_S(t, X_MC[:, 0], X_MC[:, 1])))
    return cond - marg


def n_shapley_interaction(F_S, t, x_ref, K, p=2):
    rest = [i for i in range(p) if i not in K]
    total = 0.0
    for r in range(len(rest) + 1):
        for M in combinations(rest, r):
            Mset = set(M)
            delta = 0.0
            for Lr in range(len(K) + 1):
                for L in combinations(K, Lr):
                    sign = (-1) ** (len(K) - len(L))
                    delta += sign * value_fn(F_S, t, x_ref, Mset | set(L))
            coeff = 1.0 / ((p - len(K) + 1) * comb(p - len(K), len(M)))
            total += coeff * delta
    return total


def run():
    x_ref = np.array([0.4, -0.6])
    scen = G_scenarios()

    print("=== CLAIM 1: SurvFD separates time-dependent / time-independent effects ===")
    print("(common-random-number MC; tolerance = 6x the empirical MC standard error)\n")
    n_correct = n_total = 0
    for name, (G, gt) in scen.items():
        F = lambda t, x1, x2, G=G: logh(G, t, x1, x2)
        subsets = [(0,), (1,), (0, 1)]
        verdict, _, tol = classify(F, x_ref, subsets)
        gt_str = "ambiguous (boundary case)" if gt["Iid"] is None else f"Iid={gt['Iid']} Id={gt['Id']}"
        print(f"{name:<28} ground truth: {gt_str}  (tol={tol:.2e})")
        print(f"  {'':<28} SurvFD verdict: x1={verdict[(0,)]}  x2={verdict[(1,)]}  x1x2={verdict[(0,1)]}")
        if gt["Iid"] is not None:
            exp = {M: ("TD" if M in gt["Id"] else "TI") for M in subsets}
            ok = all(verdict[M] == exp[M] for M in subsets)
            n_total += 1; n_correct += int(ok)
            print(f"  {'':<28} matches Theorem 3.2 exact-recovery prediction: {ok}")
        print()
    print(f"Theorem-3.2-covered scenarios (A,B,C) recovered exactly: {n_correct}/{n_total}\n")

    print("=== CLAIM 2: SurvSHAP-IQ (n-Shapley) detects/quantifies interactions ===")
    print("(target is S(t|x); note the exp(.) link means even additive G can leak a small")
    print(" interaction signal onto the survival scale -- the real test is MAGNITUDE, not zero)\n")
    mags = {}
    for name, (G, gt) in scen.items():
        F_S = lambda t, x1, x2, G=G: survival(G, t, x1, x2)
        vals = [n_shapley_interaction(F_S, t, x_ref, K=(0, 1)) for t in TIMES]
        mags[name] = np.mean(np.abs(vals))
        print(f"{name:<28} pairwise interaction index at t={list(TIMES)}: "
              f"[{', '.join(f'{v:+.4f}' for v in vals)}]   mean|.|={mags[name]:.4f}")
    print(f"\nNo-interaction scenarios (A,B) mean magnitude: {np.mean([mags['A_additive_TI_nointer'], mags['B_additive_TD_nointer']]):.4f}")
    print(f"Interaction scenarios (C,D) mean magnitude:    {np.mean([mags['C_linear_TI_interaction'], mags['D_nonlinear_boundary']]):.4f}")
    ratio = np.mean([mags['C_linear_TI_interaction'], mags['D_nonlinear_boundary']]) / \
            max(np.mean([mags['A_additive_TI_nointer'], mags['B_additive_TD_nointer']]), 1e-9)
    print(f"Ratio (with-interaction / without-interaction): {ratio:.2f}x")


if __name__ == "__main__":
    run()
