"""verify_claim2.py — Claim 2: Theorem 4.3 (A2-RCGP-UCB) regret bound
    R_T = Otilde( (1 + T_c Psi(T_c)^2) sqrt(beta'_T T (gamma_T + T_c)) )
gives sublinear regret for T_c = O(T^alpha) iff alpha < 1/7 (RBF case).
Same symbolic machinery as verify_claim1.py, plus the paper's intermediate form
R_T = Otilde(T_c^{7/2} sqrt(T)) and the Matern rows of Table 1.
MUTATION: drop Psi -> must give the paper's "Ideal" column alpha < 1/3.
"""
import json, sys, sympy as sp

from mpmath import mp, mpf, log, sqrt
mp.dps = 60

def _num_exponent(logb, T1=mpf(10) ** 40, T2=mpf(10) ** 80):
    """numeric growth exponent via slope of log(bound) vs log(T) (bounds are pure power laws)."""
    return float((logb(T2) - logb(T1)) / (log(T2) - log(T1)))

T, a = sp.symbols("T alpha", positive=True)


def exponent(expr):
    return sp.simplify(sp.limit(sp.log(expr) / sp.log(T), T, sp.oo))


def a2_logbound(alpha, gamma_exp=0, beta_exp=0, use_psi=True):
    def f(T):
        Tc = T ** mpf(alpha)
        psi2 = (1 + Tc * (1 + Tc)) if use_psi else mpf(1)
        return (log(1 + Tc * psi2)
                + mpf(0.5) * (mpf(beta_exp) * log(T) + log(T) + log(T ** mpf(gamma_exp) + Tc)))
    return f


def a2_exp_num(alpha, **kw):
    return _num_exponent(a2_logbound(alpha, **kw))


def a2_bound(alpha, gamma_exp=0, beta_exp=0, use_psi=True):
    Tc = T ** alpha
    Psi2 = (1 + Tc * (1 + Tc)) if use_psi else sp.Integer(1)  # Psi(T_c)^2, kappa = sn^2 = 1
    return (1 + Tc * Psi2) * sp.sqrt(T ** beta_exp * T * (T ** gamma_exp + Tc))


out = {"claim": 2, "source": "Theorem 4.3 (Sec 4.3) + Appendix D.5.5 / Table 1 (Appendix E)"}

e_rbf = exponent(a2_bound(a))
sols = sp.solve(sp.Eq(e_rbf, 1), a)
out["exponent_rbf_symbolic"] = str(e_rbf)
out["alpha_threshold_rbf"] = [str(sp.nsimplify(s)) for s in sols]
out["alpha_threshold_rbf_float"] = [float(s) for s in sols]

e_paper = exponent((T ** a) ** sp.Rational(7, 2) * sp.sqrt(T))
out["paper_intermediate_exponent"] = str(sp.simplify(e_paper))
out["intermediate_matches_bound"] = bool(sp.simplify(e_paper - e_rbf) == 0)

grid = {}
for al in [sp.Rational(1, 10), sp.Rational(1, 7), sp.Rational(1, 6), sp.Rational(1, 4), sp.Rational(1, 3)]:
    ex = a2_exp_num(float(al))
    grid[str(al)] = {"exponent": ex, "sublinear": bool(ex < 1)}
out["exponent_grid"] = grid

matern = {}
for eta_val in [sp.Rational(1, 10), sp.Rational(1, 4), sp.Rational(1, 3), sp.Rational(1, 2)]:
    for case, bexp in (("cases1_2", 0), ("case3", eta_val)):
        best = None
        for k in range(1, 7001):
            al = sp.Rational(k, 7000)
            if a2_exp_num(float(al), gamma_exp=float(eta_val), beta_exp=float(bexp)) < 1 - 1e-12:
                best = al
            else:
                break
        pred = (sp.Min(sp.Rational(1, 7), (1 - eta_val) / 6) if case == "cases1_2"
                else sp.Min((1 - eta_val) / 7, (1 - 2 * eta_val) / 6))
        matern[f"eta={eta_val}|{case}"] = {
            "empirical_sup_alpha": float(best) if best is not None else 0.0,
            "paper_table1": max(float(pred), 0.0),
            "agree": abs((float(best) if best is not None else 0.0) - max(float(pred), 0.0))
                     <= 2.0 / 7000 + 1e-9}
out["matern_table1"] = matern

e_mut = exponent(a2_bound(a, use_psi=False))
sols_mut = sp.solve(sp.Eq(e_mut, 1), a)
out["mutation_drop_Psi"] = {
    "exponent": str(e_mut), "alpha_threshold": [float(s) for s in sols_mut],
    "paper_ideal_column": 1.0 / 3,
    "matches_paper_ideal": any(abs(float(s) - 1.0 / 3) < 1e-12 for s in sols_mut),
    "verdict_changed": abs(float(sols_mut[0]) - float(sols[0])) > 1e-9}

out["claim_alpha_seventh_confirmed"] = any(abs(float(s) - 1.0 / 7) < 1e-12 for s in sols)
out["command"] = ".venv/bin/python verify_claim2.py"
json.dump(out, open("results/claim2.json", "w"), indent=2)
print(json.dumps(out, indent=2))
sys.exit(0 if out["claim_alpha_seventh_confirmed"] else 1)
