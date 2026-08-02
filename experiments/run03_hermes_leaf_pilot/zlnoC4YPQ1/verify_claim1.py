"""verify_claim1.py — Claim 1: Theorem 4.2 (FC-RCGP-UCB) regret bound
    R_T = Otilde( Psi(T_c) (1+sqrt(T_c)) sqrt(beta'_T T (gamma_T + T_c)) )
gives sublinear regret for T_c = O(T^alpha) iff alpha < 1/4 (RBF: gamma_T, beta'_T = Otilde(1)).

Method: symbolic (sympy). Substitute T_c = T^alpha and the exact Psi from Sec 4.1 into the
stated bound, extract the growth exponent e(alpha) = lim_{T->oo} log(bound)/log(T), and solve
e(alpha) < 1. Also checks the paper's intermediate form R_T = Otilde(T_c^2 sqrt(T)) and the
Matern rows of Table 1 (Appendix E).
MUTATION: remove the observability penalty Psi (set Psi=1) -> must reproduce the paper's
"Ideal" column alpha < 1/2, i.e. the verdict must change exactly as the paper predicts.
"""
import json, sys, sympy as sp

from mpmath import mp, mpf, log, sqrt
mp.dps = 60

def _num_exponent(logb, T1=mpf(10) ** 40, T2=mpf(10) ** 80):
    """numeric growth exponent via slope of log(bound) vs log(T) (bounds are pure power laws)."""
    return float((logb(T2) - logb(T1)) / (log(T2) - log(T1)))

T, a, eta = sp.symbols("T alpha eta", positive=True)


def exponent(expr):
    """growth exponent of expr in T (assumes expr ~ T^e)."""
    return sp.simplify(sp.limit(sp.log(expr) / sp.log(T), T, sp.oo))


def fc_logbound(alpha, gamma_exp=0, beta_exp=0, use_psi=True):
    """log of the Theorem 4.2 bound with T_c = T^alpha, kappa = sn^2 = 1."""
    def f(T):
        Tc = T ** mpf(alpha)
        lpsi = mpf(0.5) * log(1 + Tc * (1 + Tc)) if use_psi else mpf(0)
        return (lpsi + log(1 + sqrt(Tc))
                + mpf(0.5) * (mpf(beta_exp) * log(T) + log(T) + log(T ** mpf(gamma_exp) + Tc)))
    return f


def fc_exp_num(alpha, **kw):
    return _num_exponent(fc_logbound(alpha, **kw))


def fc_bound(alpha, gamma_exp=0, beta_exp=0, use_psi=True):
    Tc = T ** alpha
    Psi = sp.sqrt(1 + Tc * (1 + Tc)) if use_psi else sp.Integer(1)  # kappa = sn^2 = 1
    gam = T ** gamma_exp
    bet = T ** beta_exp
    return Psi * (1 + sp.sqrt(Tc)) * sp.sqrt(bet * T * (gam + Tc))


def threshold(bound_fn, **kw):
    """largest alpha with exponent < 1, solved symbolically."""
    e = exponent(bound_fn(a, **kw))
    sol = sp.solve(sp.Eq(e, 1), a)
    return e, [sp.nsimplify(s) for s in sol]


out = {"claim": 1, "source": "Theorem 4.2 (Sec 4.3) + Appendix D.4.3 / Table 1 (Appendix E)"}

# --- main result: RBF (gamma_T = Otilde(1), beta'_T = Otilde(1))
e_rbf, sols = threshold(fc_bound)
out["exponent_rbf_symbolic"] = str(e_rbf)
out["alpha_threshold_rbf"] = [str(s) for s in sols]
out["alpha_threshold_rbf_float"] = [float(s) for s in sols]

# --- paper's intermediate claim R_T = Otilde(T_c^2 sqrt(T))
e_paper = exponent((T ** a) ** 2 * sp.sqrt(T))
out["paper_intermediate_exponent"] = str(sp.simplify(e_paper))
out["intermediate_matches_bound"] = bool(sp.simplify(e_paper - e_rbf) == 0)

# --- numeric sanity: exponent evaluated at several alphas
grid = {}
for al in [sp.Rational(1, 10), sp.Rational(1, 5), sp.Rational(1, 4), sp.Rational(1, 3), sp.Rational(1, 2)]:
    ex = fc_exp_num(float(al))
    grid[str(al)] = {"exponent": ex, "sublinear": bool(ex < 1)}
out["exponent_grid"] = grid

# --- Table 1 (Appendix E) Matern rows, cases 1&2 (beta'=1) and case 3 (beta'=gamma)
matern = {}
for eta_val in [sp.Rational(1, 10), sp.Rational(1, 4), sp.Rational(1, 3), sp.Rational(1, 2)]:
    for case, bexp in (("cases1_2", 0), ("case3", eta_val)):
        best = None
        for k in range(1, 4001):  # scan alpha on a fine rational grid
            al = sp.Rational(k, 4000)
            if fc_exp_num(float(al), gamma_exp=float(eta_val), beta_exp=float(bexp)) < 1 - 1e-12:
                best = al
            else:
                break
        pred = (sp.Min(sp.Rational(1, 4), (1 - eta_val) / 3) if case == "cases1_2"
                else sp.Min((1 - eta_val) / 4, (1 - 2 * eta_val) / 3))
        matern[f"eta={eta_val}|{case}"] = {
            "empirical_sup_alpha": float(best) if best is not None else 0.0,
            "paper_table1": max(float(pred), 0.0),
            "agree": abs((float(best) if best is not None else 0.0) - max(float(pred), 0.0))
                     <= 2.0 / 4000 + 1e-9}
out["matern_table1"] = matern

# --- MUTATION: drop Psi (the observability penalty) -> paper's "Ideal" column = 1/2
e_mut, sols_mut = threshold(fc_bound, use_psi=False)
out["mutation_drop_Psi"] = {
    "exponent": str(e_mut), "alpha_threshold": [float(s) for s in sols_mut],
    "paper_ideal_column": 0.5,
    "matches_paper_ideal": any(abs(float(s) - 0.5) < 1e-12 for s in sols_mut),
    "verdict_changed": abs(float(sols_mut[0]) - float(sols[0])) > 1e-9}

out["claim_alpha_quarter_confirmed"] = any(abs(float(s) - 0.25) < 1e-12 for s in sols)
out["command"] = ".venv/bin/python verify_claim1.py"
json.dump(out, open("results/claim1.json", "w"), indent=2)
print(json.dumps(out, indent=2))
sys.exit(0 if out["claim_alpha_quarter_confirmed"] else 1)
