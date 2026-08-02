"""verify_claim2.py -- Theorem 3.4 (FDR control), Section 3.2, arXiv:2601.23124v2:

    Given nu_j and rho_j for every j,  FDR(S_SKO) <= q.

Procedure reproduced (Sec 3.2 + official src/semi_KO.py, p_val='FDR'):
  W_SKO^j = mean_i l(m(Xtilde_1^(j)_i), y_i) - mean_i l(m(Xtilde_2^(j)_i), y_i)
  T_q     = min{ t in W : (1 + #{j: W^j <= -t}) / (#{j: W^j >= t} v 1) <= q }   (Eq. 1)
  S_SKO   = { j : W^j >= T_q }

FDR is estimated as the Monte-Carlo mean of the FDP over many independent seeds
(FDP := 0 when nothing is selected, as in Benjamini-Hochberg / knockoffs).

MUTATIONS (each must break the guarantee in the predicted direction):
  M1 "no offset"  : drop the `1 +` in the numerator of Eq. (1). The +1 is exactly the
                    finite-sample correction that makes the martingale argument work, so
                    FDR must exceed q.
  M2 "asymmetric" : replace the two-sided semi-knockoff statistic by the HRT-style
                    l(m(Xtilde_1)) - l(m(X)) computed on the training data. Lemma 2.1's
                    coin-flip sign symmetry is then violated -> FDR must exceed q.
  M3 "rho := nu"  : use nu_j on BOTH sides. Signs stay symmetric (so FDR should stay
                    controlled) but the alternative is no longer detectable -> power ~ 0.
                    This is a *negative control* on the mutation: it isolates which
                    ingredient buys FDR control and which buys power.

Usage: ./.venv/bin/python verify_claim2.py
"""
import json, time
import numpy as np
from common import (gen_adjacent, oracle_nu, oracle_rho, ridge_fit, ridge_pred,
                    sko_pair_losses, hrt_style_losses, knockoff_threshold, fdp)

N, P = 300, 20
N_PERM = 5
QS = [0.05, 0.10, 0.20]
N_SEEDS = 300


def stats_for_seed(seed, mode):
    rng = np.random.default_rng(50_000 + seed)
    X, y, beta, Sigma, sn = gen_adjacent(N, P, rng)
    non_null = set(np.where(beta != 0)[0].tolist())
    theta = ridge_fit(X, y)
    pred = lambda Z: ridge_pred(theta, Z)

    W = np.zeros(P)
    for j in range(P):
        nu = oracle_nu(X, Sigma, j)
        if mode == "asymmetric":
            La, Lb = hrt_style_losses(X, y, j, nu, pred, rng, N_PERM)
        elif mode == "rho_eq_nu":
            La, Lb = sko_pair_losses(X, y, j, nu, nu, pred, rng, N_PERM)
        else:
            rho = oracle_rho(X, y, Sigma, beta, sn, j)
            La, Lb = sko_pair_losses(X, y, j, nu, rho, pred, rng, N_PERM)
        W[j] = La.mean() - Lb.mean()
    return W, non_null


def evaluate(mode, offset):
    res = {q: {"fdp": [], "power": [], "n_sel": []} for q in QS}
    for s in range(N_SEEDS):
        W, non_null = stats_for_seed(s, mode)
        for q in QS:
            t = knockoff_threshold(W, q, offset=offset)
            sel = np.where(W >= t)[0] if np.isfinite(t) else np.array([], dtype=int)
            res[q]["fdp"].append(fdp(sel.tolist(), non_null))
            res[q]["power"].append(len(set(sel.tolist()) & non_null) / max(len(non_null), 1))
            res[q]["n_sel"].append(len(sel))
    return {str(q): {"FDR": float(np.mean(v["fdp"])),
                     "FDR_mcse": float(np.std(v["fdp"]) / np.sqrt(N_SEEDS)),
                     "power": float(np.mean(v["power"])),
                     "mean_n_selected": float(np.mean(v["n_sel"]))}
            for q, v in res.items()}


if __name__ == "__main__":
    t0 = time.time()
    out = {
        "claim": 2,
        "source": "Theorem 3.4 (FDR control), Section 3.2; threshold Eq. (1), Section 2.3.2",
        "command": "./.venv/bin/python verify_claim2.py",
        "config": {"n": N, "p": P, "n_perm": N_PERM, "n_seeds": N_SEEDS, "q_levels": QS,
                   "n_non_null": 5,
                   "design": "adjacent support, Sec 5.1 / App F.5.1",
                   "black_box": "ridge fitted on all n samples (no split)",
                   "nu,rho": "ORACLE closed form (Theorem 3.4 hypothesis)"},
        "MAIN_semi_knockoffs": evaluate("sko", offset=1),
        "M1_mutation_no_plus_one_offset": evaluate("sko", offset=0),
        "M2_mutation_asymmetric_statistic": evaluate("asymmetric", offset=1),
        "M3_negative_control_rho_equals_nu": evaluate("rho_eq_nu", offset=1),
        "predictions": {
            "MAIN": "FDR <= q at every q",
            "M1": "FDR > q (the +1 offset is the finite-sample correction)",
            "M2": "FDR > q (sign exchangeability of Lemma 2.1 destroyed)",
            "M3": "FDR still <= q but power collapses to ~0"},
    }
    out["wall_seconds"] = round(time.time() - t0, 2)
    with open("results/claim2.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
