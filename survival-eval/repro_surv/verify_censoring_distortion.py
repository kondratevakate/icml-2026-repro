"""
Reproduction of the two claims of:
  "When Can We Trust Survival Model Evaluation ?"
  Bahrini, Razakarivony, Dupuy, Gares, Barbet-Massin. ICML 2026 (orid Y9gsOEdaNE).
  Authors' code: https://github.com/Ghanem01/When-Can-We-Trust-Survival-Model-Evaluation

Claim 1: censoring induces SYSTEMATIC, MECHANISM-DEPENDENT distortions in survival evaluation
         metrics, varying with censoring rate and censoring mechanism.
Claim 2: moderate numerical bias can make MODEL COMPARISON unreliable as censoring increases.

Design (following the paper's ST-vs-OR logic): the same fixed models and the same predictions are
scored twice -- ST ("standard": metric computed on censored observations) and OR ("oracle": metric
computed on the fully observed true event times). Only the test outcomes differ, so the ST-OR gap
isolates the distortion the censoring introduces into the metric itself, with no retraining effect.

Fully controlled synthetic version (CPU, numpy only, no external data): true event times are known
by construction, so the oracle is exact. Censoring mechanisms follow the paper:
  - administrative       : C = tau (fixed end of study)
  - independent          : C ~ Exp(rate), independent of covariates
  - covariate-dependent  : C ~ Exp(rate * exp(gamma'X)), depends on covariates
Censoring rate is tuned by bisection to hit each target level.

Metrics: Harrell's C-index (concordance family) and the Integrated Brier Score with IPCW
censoring weights (IBS family) -- the two families the paper highlights.
"""
import numpy as np

rng_global = np.random.default_rng(0)
P = 5
BETA = np.array([1.0, -0.8, 0.6, 0.0, 0.0])


# ------------------------------- data generating process --------------------------------
def gen_data(n, rng):
    X = rng.standard_normal((n, P))
    lp = X @ BETA                       # true linear predictor
    lam = 0.1 * np.exp(lp)              # Cox-exponential hazard
    T = rng.exponential(1.0 / lam)      # true event time (always known here = oracle)
    return X, T, lp


def censor(X, T, mech, level, rng):
    """Return observed time, event indicator. `level` = target censoring rate in [0,1)."""
    n = len(T)
    if level <= 0:
        return T.copy(), np.ones(n, bool)

    def make(par):
        if mech == "administrative":
            C = np.full(n, par)
        elif mech == "independent":
            C = rng.exponential(par, size=n)
        elif mech == "covariate-dependent":
            gamma = np.array([0.9, 0.0, -0.7, 0.0, 0.0])
            C = rng.exponential(par * np.exp(-(X @ gamma)), size=n)
        else:
            raise ValueError(mech)
        return C

    lo, hi = 1e-3, 1e4                  # bisection on the scale parameter to hit `level`
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        rate = float((make(mid) < T).mean())
        if rate > level: lo = mid
        else: hi = mid
    C = make(np.sqrt(lo * hi))
    obs = np.minimum(T, C)
    ev = T <= C
    return obs, ev


# ------------------------------------- metrics ------------------------------------------
def c_index(risk, time, event):
    """Harrell's C-index. With event=all-True this is the oracle (uncensored) concordance."""
    order = np.argsort(time)
    t, e, r = time[order], event[order], risk[order]
    num = den = 0.0
    for i in range(len(t)):
        if not e[i]:
            continue                     # only pairs whose earlier member is an event are usable
        later = t > t[i]
        if not later.any():
            continue
        den += later.sum()
        num += (r[later] < r[i]).sum() + 0.5 * (r[later] == r[i]).sum()
    return num / den if den else np.nan


def km_censoring(time, event):
    """Kaplan-Meier estimate of the CENSORING survival G(t) (reverse KM)."""
    order = np.argsort(time)
    t, e = time[order], event[order]
    uniq = np.unique(t)
    G, surv, n = {}, 1.0, len(t)
    at_risk = n
    for u in uniq:
        d_c = int(((t == u) & (~e)).sum())     # censoring events at u
        if at_risk > 0 and d_c > 0:
            surv *= (1 - d_c / at_risk)
        G[u] = surv
        at_risk -= int((t == u).sum())
    ts, gs = np.array(list(G.keys())), np.array(list(G.values()))
    def Gfun(x):
        idx = np.searchsorted(ts, x, side="right") - 1
        return np.where(idx >= 0, gs[np.clip(idx, 0, len(gs) - 1)], 1.0)
    return Gfun


def ibs(risk, time, event, grid, oracle=False):
    """Integrated Brier Score. Survival predicted as exp(-lam0*t*exp(risk)) (proportional hazards).
    ST version uses IPCW weights from the censoring KM; OR version uses the true event times."""
    S = lambda t: np.exp(-0.1 * t * np.exp(risk))
    if oracle:
        errs = [np.mean((( time > t).astype(float) - S(t)) ** 2) for t in grid]
        return float(np.mean(errs))
    G = km_censoring(time, event)
    errs = []
    for t in grid:
        st = S(t)
        alive = time > t
        died = (time <= t) & event
        g_t = np.maximum(G(np.array([t]))[0], 1e-6)
        g_ti = np.maximum(G(time), 1e-6)
        w = np.where(alive, 1.0 / g_t, np.where(died, 1.0 / g_ti, 0.0))
        errs.append(np.mean(w * (alive.astype(float) - st) ** 2))
    return float(np.mean(errs))


# ------------------------------------ experiment ----------------------------------------
def models(X, lp, rng, close=False):
    """Fixed risk scores -> a known oracle ranking.
    close=False: widely separated models (ranking is robust by construction).
    close=True : NEAR-TIED models, which is the regime the paper's ranking-instability claim is
                 about (their rebuttal notes that where rankings swap, the top models are often
                 not statistically distinguishable). Small perturbations of the correct score."""
    if not close:
        return {
            "M1 correct":       lp,
            "M2 partial":       X[:, :2] @ BETA[:2],
            "M3 noisy":         lp + rng.standard_normal(len(lp)) * 1.2,
            "M4 uninformative": rng.standard_normal(len(lp)),
        }
    return {
        "A":  lp + rng.standard_normal(len(lp)) * 0.55,
        "B":  lp + rng.standard_normal(len(lp)) * 0.60,
        "C":  lp + rng.standard_normal(len(lp)) * 0.65,
        "D":  lp + rng.standard_normal(len(lp)) * 0.70,
    }


def run(n=4000, reps=8, levels=(0.0, 0.2, 0.4, 0.6, 0.8),
        mechs=("administrative", "independent", "covariate-dependent")):
    print(f"n={n} per rep, reps={reps}\n")
    print("CLAIM 1 - metric bias (ST minus OR), by mechanism and censoring rate")
    print(f"{'mechanism':<22}{'rate':>6} | {'C-index ST':>11}{'OR':>8}{'bias':>8} | {'IBS ST':>9}{'OR':>8}{'bias':>8}")
    rank_rows = []
    for mech in mechs:
        for lv in levels:
            cs, co, bs, bo, agree = [], [], [], [], []
            for r in range(reps):
                rng = np.random.default_rng(100 + r)
                X, T, lp = gen_data(n, rng)
                obs, ev = censor(X, T, mech, lv, rng)
                M = models(X, lp, rng)
                grid = np.quantile(T, [0.2, 0.4, 0.6, 0.8])
                st_c = {k: c_index(v, obs, ev) for k, v in M.items()}
                or_c = {k: c_index(v, T, np.ones(n, bool)) for k, v in M.items()}
                st_b = {k: ibs(v, obs, ev, grid) for k, v in M.items()}
                or_b = {k: ibs(v, T, np.ones(n, bool), grid, oracle=True) for k, v in M.items()}
                cs.append(np.mean(list(st_c.values()))); co.append(np.mean(list(or_c.values())))
                bs.append(np.mean(list(st_b.values()))); bo.append(np.mean(list(or_b.values())))
                # ranking preservation (higher C = better; lower IBS = better)
                rk_st = sorted(st_c, key=lambda k: -st_c[k])
                rk_or = sorted(or_c, key=lambda k: -or_c[k])
                agree.append(rk_st == rk_or)
            c_s, c_o, b_s, b_o = np.mean(cs), np.mean(co), np.mean(bs), np.mean(bo)
            print(f"{mech:<22}{lv:>6.0%} | {c_s:>11.4f}{c_o:>8.4f}{c_s-c_o:>+8.4f} | "
                  f"{b_s:>9.4f}{b_o:>8.4f}{b_s-b_o:>+8.4f}")
            rank_rows.append((mech, lv, float(np.mean(agree))))
        print()

    print("CLAIM 2a - ranking preservation, WIDELY SEPARATED models (robust by construction)")
    print(f"{'mechanism':<22}" + "".join(f"{l:>9.0%}" for l in levels))
    for mech in mechs:
        vals = [a for (m, l, a) in rank_rows if m == mech]
        print(f"{mech:<22}" + "".join(f"{v:>9.2f}" for v in vals))

    print("\nCLAIM 2b - ranking preservation, NEAR-TIED models (the regime the claim is about)")
    print(f"{'mechanism':<22}" + "".join(f"{l:>9.0%}" for l in levels))
    for mech in mechs:
        row = []
        for lv in levels:
            agree = []
            for r in range(reps * 3):
                rng = np.random.default_rng(500 + r)
                X, T, lp = gen_data(n, rng)
                obs, ev = censor(X, T, mech, lv, rng)
                M = models(X, lp, rng, close=True)
                st = {k: c_index(v, obs, ev) for k, v in M.items()}
                orc = {k: c_index(v, T, np.ones(n, bool)) for k, v in M.items()}
                agree.append(sorted(st, key=lambda k: -st[k]) == sorted(orc, key=lambda k: -orc[k]))
            row.append(float(np.mean(agree)))
        print(f"{mech:<22}" + "".join(f"{v:>9.2f}" for v in row))


if __name__ == "__main__":
    run()
