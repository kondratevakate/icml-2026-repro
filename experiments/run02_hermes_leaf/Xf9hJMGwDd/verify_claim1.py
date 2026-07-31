"""verify_claim1.py -- Theorem 3.3 (Sec 3.1): the nonparametric paired-test Semi-knockoffs
procedure yields VALID p-values for any pre-trained model, with NO train-test split,
requiring only the conditional expectations nu_j and rho_j.

Operationalisation: a p-value is valid iff it is super-uniform under H0, i.e.
    P(p <= alpha) <= alpha  for all alpha.
We estimate P(p<=alpha) over many independent seeds, for null coordinates j (beta_j = 0),
with the black box m_hat (ridge) fitted on ALL the data (no split) -- exactly the regime
where a naive statistic would be invalid.

Three arms:
  A. ORACLE   : nu_j, rho_j in closed form            -> Theorem 3.3 hypothesis exactly.
  B. ESTIMATED: nu_hat, rho_hat fitted by ridge       -> Section 4.1 / Algorithm 1.
  C. MUTATION : symmetric pairing REPLACED by the HRT-style asymmetric statistic
                l(m(Xtilde_1)) - l(m(X)) on the training data (Sec 3, paragraph
                "independence between the test set and the model is required").
                Prediction: type-I error must blow far above alpha.

Usage: ./.venv/bin/python verify_claim1.py
"""
import json, sys, time
import numpy as np
from common import (gen_adjacent, oracle_nu, oracle_rho, ridge_fit, ridge_pred,
                    sko_pair_losses, hrt_style_losses, sign_test_pval, wilcoxon_pval)

N_SEEDS = 600
N, P = 200, 10
ALPHAS = [0.05, 0.10, 0.20]


def ridge_impute(A, b, lam=1e-2):
    th = ridge_fit(A, b, lam)
    return ridge_pred(th, A)


def run():
    p_oracle_sign, p_oracle_wcx = [], []
    p_est_sign, p_est_wcx = [], []
    p_mut_sign, p_mut_wcx = [], []

    for s in range(N_SEEDS):
        rng = np.random.default_rng(10_000 + s)
        X, y, beta, Sigma, sn = gen_adjacent(N, P, rng)
        nulls = np.where(beta == 0)[0]
        j = int(rng.choice(nulls))                      # a genuinely null coordinate

        theta = ridge_fit(X, y)                          # black box, NO train-test split
        pred = lambda Z: ridge_pred(theta, Z)

        # --- A. oracle nu, rho
        nu = oracle_nu(X, Sigma, j)
        rho = oracle_rho(X, y, Sigma, beta, sn, j)
        L1, L2 = sko_pair_losses(X, y, j, nu, rho, pred, rng)
        p_oracle_sign.append(sign_test_pval(L1, L2))
        p_oracle_wcx.append(wilcoxon_pval(L1, L2))

        # --- B. estimated nu_hat, rho_hat (Algorithm 1)
        idx = np.delete(np.arange(P), j)
        nu_h = ridge_impute(X[:, idx], X[:, j])
        rho_h = ridge_impute(np.column_stack([X[:, idx], y]), X[:, j])
        L1, L2 = sko_pair_losses(X, y, j, nu_h, rho_h, pred, rng)
        p_est_sign.append(sign_test_pval(L1, L2))
        p_est_wcx.append(wilcoxon_pval(L1, L2))

        # --- C. MUTATION: asymmetric HRT-style statistic, no split
        La, Lb = hrt_style_losses(X, y, j, nu, pred, rng)
        p_mut_sign.append(sign_test_pval(La, Lb))
        p_mut_wcx.append(wilcoxon_pval(La, Lb))

    def rates(pv):
        pv = np.asarray(pv)
        return {f"alpha={a}": float(np.mean(pv <= a)) for a in ALPHAS}

    def mcse(a):  # binomial Monte-Carlo standard error
        return float(np.sqrt(a * (1 - a) / N_SEEDS))

    out = {
        "claim": 1,
        "source": "Theorem 3.3 (Type-I error), Section 3.1, arXiv:2601.23124v2",
        "command": "./.venv/bin/python verify_claim1.py",
        "config": {"n_seeds": N_SEEDS, "n": N, "p": P, "design": "X~N(0,0.6^|i-j|), y=beta'X+eps, "
                   "beta[:0.25p] ~ U[1,2] (adjacent support, Sec 5.1)",
                   "black_box": "ridge, fitted on ALL n samples (no train-test split)",
                   "alphas": ALPHAS},
        "mc_standard_errors": {f"alpha={a}": mcse(a) for a in ALPHAS},
        "A_oracle": {"sign_test": rates(p_oracle_sign), "wilcoxon": rates(p_oracle_wcx)},
        "B_estimated": {"sign_test": rates(p_est_sign), "wilcoxon": rates(p_est_wcx)},
        "C_mutation_asymmetric_HRT_style_no_split": {
            "sign_test": rates(p_mut_sign), "wilcoxon": rates(p_mut_wcx),
            "prediction": "type-I error >> alpha because exchangeability is destroyed"},
        "mean_pvalue_oracle_sign": float(np.mean(p_oracle_sign)),
        "mean_pvalue_mutation_sign": float(np.mean(p_mut_sign)),
    }
    return out


if __name__ == "__main__":
    t0 = time.time()
    out = run()
    out["wall_seconds"] = round(time.time() - t0, 2)
    with open("results/claim1.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
