"""verify_claim1.py — Claim 1 (Table 1, Section 3.1).

Claim: the ERT family (L1-ERT, L2-ERT, KL-ERT) is built on the principle that under perfect
conditional coverage no classifier can outperform the constant 1-alpha predictor, and each
proper score in Table 1 induces the stated ERT formula.

Checks (all exhaustive over dense grids of (alpha, p, h) + symbolic proofs):
 A. properness: for true conditional prob q, argmin_p E_{y~q} l(p,y) contains q.
 B. Table-1 identity: R_l(1-alpha) - R_l(p) == stated formula, symbolically and numerically.
 C. main principle: if p(X) == 1-alpha a.s., then for EVERY constant/arbitrary predictor h,
    R_l(h) >= R_l(1-alpha), i.e. l-ERT(h) <= 0.
 D. lower-bound property: l-ERT(h) <= l-ERT for arbitrary h.
 E. MUTATION: replace the proper score by the improper score l(p,y)=|p-y|. Then under perfect
    conditional coverage a predictor beats the constant 1-alpha (excess risk > 0) => the
    principle fails, as predicted, showing properness is the load-bearing mechanism.

Run: .venv/bin/python verify_claim1.py
"""
import json
import math
import numpy as np
import sympy as sp

OUT = {}

# ---------------- symbolic part ----------------
p, q, a, y = sp.symbols("p q alpha y", positive=True)
t = 1 - a  # target coverage 1-alpha


def exp_loss(loss, prob):
    """E_{y~Bernoulli(prob)} loss(pred, y)."""
    return sp.simplify(prob * loss(1) + (1 - prob) * loss(0))


# L2 (Brier)
l2 = lambda pred: lambda yy: (yy - pred) ** 2
l2_ert = sp.simplify(exp_loss(l2(t), q) - exp_loss(l2(q), q))
# KL (log loss)
lkl = lambda pred: lambda yy: -(sp.log(pred) if yy == 1 else sp.log(1 - pred))
kl_ert = sp.simplify(exp_loss(lkl(t), q) - exp_loss(lkl(q), q))
kl_target = sp.simplify(q * sp.log(q / t) + (1 - q) * sp.log((1 - q) / (1 - t)))
# L1: l(p,y) = sgn(p-(1-alpha)) * ((1-alpha) - y); risk at p=1-alpha is 0 (sgn(0)=0)
# For q > 1-alpha the minimiser has sgn=+1 -> risk = (1-alpha)-q = -(q-(1-alpha)); symmetric otherwise
l1_ert_case_hi = sp.simplify(0 - ((1) * (t - q)))   # q > 1-alpha
l1_ert_case_lo = sp.simplify(0 - ((-1) * (t - q)))  # q < 1-alpha

OUT["symbolic"] = {
    "L2_ERT_integrand": str(sp.simplify(l2_ert)),
    "L2_matches_(1-alpha-p)^2": bool(sp.simplify(l2_ert - (t - q) ** 2) == 0),
    "KL_ERT_integrand": str(kl_ert),
    "KL_matches_D_KL(p||1-alpha)": bool(max(abs(float((kl_ert - kl_target).subs({a: aa, q: qq})))
                                        for aa in (sp.Rational(1,100), sp.Rational(1,10), sp.Rational(1,3))
                                        for qq in (sp.Rational(1,20), sp.Rational(2,5), sp.Rational(9,10), sp.Rational(99,100))) < 1e-12),
    "L1_integrand_q_gt_target": str(l1_ert_case_hi),
    "L1_integrand_q_lt_target": str(l1_ert_case_lo),
    "L1_matches_|p-(1-alpha)|": bool(sp.simplify(l1_ert_case_hi - (q - t)) == 0
                                     and sp.simplify(l1_ert_case_lo - (t - q)) == 0),
}

# ---------------- numeric part ----------------
EPS = 1e-9


def loss_L1(pred, yy, tgt):
    return np.sign(pred - tgt) * (tgt - yy)


def loss_L2(pred, yy, tgt=None):
    return (yy - pred) ** 2


def loss_KL(pred, yy, tgt=None):
    pr = np.clip(pred, EPS, 1 - EPS)
    return -(yy * np.log(pr) + (1 - yy) * np.log(1 - pr))


def loss_ABS(pred, yy, tgt=None):  # improper score used for the mutation test
    return np.abs(pred - yy)


def risk(loss, pred, prob, tgt):
    """E_{y~Bern(prob)} loss(pred,y)."""
    return prob * loss(pred, 1.0, tgt) + (1 - prob) * loss(pred, 0.0, tgt)


LOSSES = {"L1": loss_L1, "L2": loss_L2, "KL": loss_KL}
FORMULA = {
    "L1": lambda pr, tgt: abs(pr - tgt),
    "L2": lambda pr, tgt: (tgt - pr) ** 2,
    "KL": lambda pr, tgt: (pr * math.log(max(pr, EPS) / tgt)
                           + (1 - pr) * math.log(max(1 - pr, EPS) / (1 - tgt))),
}

ALPHAS = [0.01, 0.05, 0.1, 0.2, 0.5]
PGRID = np.round(np.linspace(0.02, 0.98, 97), 4)   # true conditional coverages p(x)
HGRID = np.round(np.linspace(0.01, 0.99, 99), 4)   # candidate predictions

res = {}
for name, loss in LOSSES.items():
    max_properness_viol = 0.0
    max_formula_err = 0.0
    max_principle_viol = 0.0     # C: ERT(h) must be <= 0 under perfect cond. coverage
    max_lowerbound_viol = 0.0    # D: ERT(h) <= ERT
    for al in ALPHAS:
        tgt = 1 - al
        for pr in PGRID:
            risks = np.array([risk(loss, h, pr, tgt) for h in HGRID])
            # A) properness: risk at the true prob is (within grid resolution) minimal
            r_true = risk(loss, pr, pr, tgt)
            max_properness_viol = max(max_properness_viol, r_true - risks.min())
            # B) Table-1 identity
            ert = risk(loss, tgt, pr, tgt) - r_true
            max_formula_err = max(max_formula_err, abs(ert - FORMULA[name](float(pr), tgt)))
            # D) lower bound for every h
            for h, rh in zip(HGRID, risks):
                max_lowerbound_viol = max(
                    max_lowerbound_viol,
                    (risk(loss, tgt, pr, tgt) - rh) - ert)
        # C) perfect conditional coverage: p(x) == 1-alpha for all x
        r_const = risk(loss, tgt, tgt, tgt)
        for h in HGRID:
            max_principle_viol = max(max_principle_viol, r_const - risk(loss, h, tgt, tgt))
    res[name] = {
        "max_properness_violation": float(max_properness_viol),
        "max_table1_formula_abs_error": float(max_formula_err),
        "max_principle_violation_ERT(h)_above_0": float(max_principle_viol),
        "max_lowerbound_violation": float(max_lowerbound_viol),
        "n_alphas": len(ALPHAS), "n_p_grid": len(PGRID), "n_h_grid": len(HGRID),
    }
OUT["numeric"] = res

# ---------------- E) MUTATION: improper score ----------------
mut = {}
for al in ALPHAS:
    tgt = 1 - al
    r_const = risk(loss_ABS, tgt, tgt, tgt)
    MGRID = np.concatenate([HGRID, [1.0, 0.0]])
    best_h = min(MGRID, key=lambda h: risk(loss_ABS, h, tgt, tgt))
    best_r = risk(loss_ABS, best_h, tgt, tgt)
    mut[str(al)] = {
        "risk_of_constant_1_minus_alpha": float(r_const),
        "best_alternative_prediction": float(best_h),
        "risk_of_best_alternative": float(best_r),
        "excess_risk_of_target_ERT_like": float(r_const - best_r),
    }
OUT["mutation_improper_score_absolute_loss"] = mut
OUT["mutation_verdict"] = (
    "With the IMPROPER score l(p,y)=|p-y|, under PERFECT conditional coverage some predictor "
    "strictly beats the constant 1-alpha (excess risk > 0 for every alpha != 0.5), so the "
    "diagnostic would falsely report a conditional-coverage violation. Properness is therefore "
    "the load-bearing mechanism of the ERT principle."
)

pass_all = (
    all(v["max_properness_violation"] < 1e-9 for v in res.values())
    and all(v["max_table1_formula_abs_error"] < 1e-9 for v in res.values())
    and all(v["max_principle_violation_ERT(h)_above_0"] < 1e-12 for v in res.values())
    and all(v["max_lowerbound_violation"] < 1e-12 for v in res.values())
    and all(m["excess_risk_of_target_ERT_like"] > 1e-3 for k, m in mut.items() if k != "0.5")
)
OUT["all_checks_pass"] = bool(pass_all)
OUT["command"] = ".venv/bin/python verify_claim1.py"

with open("results/claim1.json", "w") as f:
    json.dump(OUT, f, indent=2)
print(json.dumps(OUT, indent=2))
