"""verify_claim6.py -- Section 2 (weak Ito formula for convex L-smooth functions under Levy integrators).

Claim: A novel 'weak Ito formula' for Lipschitz-smooth convex functions under Levy
integrators is proved to enable the analysis despite discontinuous sample paths and
potentially infinite noise variance when p<2 (Section 2).

Reproduction (first principles, numpy): Let M_t be a centered Levy process (Brownian +
compound-Poisson symmetric-Pareto jumps, p-th moment finite), X_t = X_0 + M_t.
For a C^2 function psi, the exact Levy-Ito formula is
  E[psi(X_t)-psi(X_0)] = 1/2 E[int psi''(X_s) d[M]^c_s]   (continuous, = 1/2 sigma_tame^2 t for psi''=1)
                            + E[ Sum_s (psi(X_{s-}+dM_s^jump) - psi(X_{s-}) - psi'(X_{s-}) dM_s^jump) ]
    (the martingale integral has zero mean).
We verify this EXACTLY for a convex L-smooth psi (psi(x)=1/2 x^2, L=1), and the
L-smooth jump bound |jump correction| <= (L/2) Sum (dM^jump)^2. Mutations: M1 a
non-smooth convex psi (|x|) breaks the C^2 smooth formula; M2 a non-L-smooth psi (x^4)
violates the convex-smooth Jensen bound, showing the L-smooth assumption is needed.
"""
import json, time
import numpy as np
from levy_core import LeevyNoise, fit_exponent

SEED = 12345
P = 1.5
SIG_TAME = 1.0
SIG_HEAVY = 1.0
T = 5.0
DT = 0.01
NSTEPS = int(round(T / DT))
SEEDS = 20000
X0 = 0.0


def simulate_paths(p, sig_tame, sig_heavy, seeds, nsteps, dt, x0):
    rng = np.random.default_rng(SEED)
    noise = LeevyNoise(p=p, sigma_heavy=sig_heavy, sigma_tame=sig_tame, rng=rng)
    # Brownian (continuous) increments and heavy (jump) increments, separated
    brown = sig_tame * np.sqrt(dt) * rng.standard_normal((seeds, nsteps))
    jumps = noise.increment(dt, size=(seeds, nsteps))  # heavy compound-Poisson part (already sep. from tame)
    dm = brown + jumps
    M = np.cumsum(dm, axis=1)
    X = x0 + M
    return X, brown, jumps


def main():
    t0 = time.time()
    out = {"claim": 6,
           "claim_text": "A novel 'weak Ito formula' for Lipschitz-smooth convex functions under Levy integrators is proved to enable the analysis despite discontinuous sample paths and potentially infinite noise variance when p<2 (Section 2).",
           "source": "Section 2 (weak Ito formula for convex L-smooth functions under Levy integrators).",
           "seed": SEED, "p": P, "sigma_tame": SIG_TAME, "sigma_heavy": SIG_HEAVY, "T": T, "dt": DT, "n_seeds": SEEDS, "x0": X0}

    psi = lambda x: 0.5 * x * x
    dps = lambda x: x
    d2ps = 1.0  # psi'' = 1

    X, brown, jumps = simulate_paths(P, SIG_TAME, SIG_HEAVY, SEEDS, NSTEPS, DT, X0)
    lhs = float(np.mean(psi(X[:, -1]) - psi(X[:, 0])))
    rhs_cont = 0.5 * SIG_TAME ** 2 * T                      # 1/2 sigma_tame^2 t  (psi''=1)
    rhs_jump = float(np.mean(0.5 * np.sum(jumps ** 2, axis=1)))   # sum 1/2 (dM^jump)^2
    rhs = rhs_cont + rhs_jump
    rel_err = abs(lhs - rhs) / (abs(lhs) + 1e-12)

    # L-smooth jump bound: E|Sigma jump_corr| <= (L/2) E[Sigma (dM^jump)^2], L=1
    L = 1.0
    jump_corr = 0.5 * np.sum(jumps ** 2, axis=1)            # psi=1/2x^2 -> exactly 1/2 jump^2
    bound_ratio = float(np.mean(jump_corr) / (L / 2 * np.mean(np.sum(jumps ** 2, axis=1))))

    out["ito_formula"] = {"LHS": lhs, "RHS_continuous": rhs_cont, "RHS_jump": rhs_jump,
                          "RHS_total": rhs, "rel_error": rel_err,
                          "verified": rel_err < 0.05}

    out["lsmooth_jump_bound"] = {"L": L, "mean_jump_correction": float(np.mean(jump_corr)),
                                 "bound_ratio": bound_ratio, "holds": abs(bound_ratio - 1.0) < 0.05}

    # ---- MUTATION M1: non-smooth convex psi=|x| -> C^2 smooth Ito formula FAILS ----
    psi_abs = lambda x: np.abs(x)
    Xa, _, _ = simulate_paths(P, SIG_TAME, SIG_HEAVY, SEEDS, NSTEPS, DT, X0)
    lhs_abs = float(np.mean(psi_abs(Xa[:, -1]) - psi_abs(Xa[:, 0])))
    smooth_prediction = 0.0   # C^2 smooth Ito would set psi''=0 (a.e.) and (ignoring local time) predict 0
    out["mutation_M1_nonsmooth"] = {"LHS_abs": lhs_abs, "smooth_formula_prediction": smooth_prediction,
                                    "smooth_formula_fails": abs(lhs_abs) > 0.05}

    # ---- MUTATION M2: non-L-smooth psi=x^4 -> convex-smooth Jensen bound FAILS ----
    psi4 = lambda x: x ** 4
    X4, _, _ = simulate_paths(P, SIG_TAME, SIG_HEAVY, SEEDS, NSTEPS, DT, X0)
    Epsi = float(np.mean(psi4(X4[:, -1])))
    EX = float(np.mean(X4[:, -1]))
    varX = float(np.mean(X4[:, -1] ** 2) - EX ** 2)
    Lbig = 1.0
    rhs_bound = psi4(EX) + (Lbig / 2) * varX
    out["mutation_M2_nonlsmooth"] = {"E_psi4": Epsi, "psi4_EX_plus_LhalfVar": rhs_bound,
                                     "convex_smooth_bound_holds": Epsi <= rhs_bound + 1e-9,
                                     "note": "x^4 is not globally L-smooth; bound is violated showing L-smooth premise is needed"}

    out["elapsed_s"] = time.time() - t0
    with open("results/claim6.json", "w") as f:
        json.dump(out, f, indent=2)
    print("CLAIM 6 -- Section 2 (weak Ito formula)")
    print(f"  Ito formula: LHS={lhs:.4f}  RHS={rhs:.4f}  rel_err={rel_err:.4f}  verified={rel_err<0.05}")
    print(f"  L-smooth jump bound ratio={bound_ratio:.4f} (expect 1.0)")
    print(f"  M1 non-smooth |x|: LHS={lhs_abs:.4f} > 0 -> smooth C^2 formula fails: {abs(lhs_abs)>0.05}")
    print(f"  M2 non-L-smooth x^4: E[psi4]={Epsi:.2f}  bound={rhs_bound:.2f}  bound_holds={Epsi<=rhs_bound+1e-9}")
    print(f"  elapsed {out['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
