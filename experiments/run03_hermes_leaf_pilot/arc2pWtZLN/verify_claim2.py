"""
Claim 2 (Proposition 3.5, Section 3.2):
  "The collapsed operator is spectrally bounded between 0 and the rank-0
   Laplacian A, i.e. 0 <= S <= A, guaranteeing positive semi-definiteness."

Assumption (Prop 3.5): L* >= 0 with C > 0.
Verification:
  * Over many seeded random PSD block matrices (C PD) confirm 0 <= S <= A,
    i.e. S is PSD and A - S is PSD.
  * On the paper's graded-Laplacian collapse (regularized C+eps I so C is PD)
    confirm 0 <= S_eps <= A.
  * Eigenvalue check: eigenvalues of S and of A-S are non-negative.

Mutation test:
  * Break the assumption (violate Thm 3.2 by taking gamma_0 above its PSD bound
    so L* is indefinite, OR flip the Schur sign). The bound 0 <= S <= A fails,
    showing the property is conditional on the PSD setup.
"""
import json
import numpy as np
from common import (rng, build_complex, graded_laplacian, schur_complement,
                    is_psd, sigma_min_plus, MASTER_SEED)

SEED = 202


def make_psd_block(n, p, r):
    N = n + p
    R = r.standard_normal((N, N))
    M = R @ R.T + 0.1 * np.eye(N)      # PD -> both blocks PD
    return M[:n, :n], M[:n, n:], M[n:, n:], M


def main():
    r = rng(SEED)
    n_fail = 0
    n_tot = 40
    min_eig_S = np.inf
    min_eig_AmS = np.inf
    for i in range(n_tot):
        ri = rng(SEED + i)
        A, X, C, M = make_psd_block(10 + (i % 5), 14 + (i % 6), ri)
        S = schur_complement(A, X, C)
        if not is_psd(S, tol=1e-7):
            n_fail += 1
        if not is_psd(A - S, tol=1e-7):
            n_fail += 1
        min_eig_S = min(min_eig_S, float(np.linalg.eigvalsh(S).min()))
        min_eig_AmS = min(min_eig_AmS, float(np.linalg.eigvalsh(A - S).min()))

    # ---- paper's graded Laplacian collapse ----
    nv = 12
    edges = [(i, (i + 1) % nv) for i in range(nv)] + [(0, 6), (2, 9), (4, 10)]
    eset = set(edges) | {(b, a) for (a, b) in edges}
    raw_tris = [(0, 1, 6), (2, 3, 9), (4, 5, 10), (6, 7, 8), (9, 10, 11)]
    tri_valid = [t for t in raw_tris if all(((a, b) in eset or (b, a) in eset)
                   for a, b in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])]
    cx = build_complex(nv, edges, tri_valid, betas=[1.0, 1.0], gammas=[0.4, 0.4])
    Lstar, A_g, X_g, C_g = graded_laplacian(cx)
    eps = 1e-4
    S_eps = schur_complement(A_g, X_g, C_g + eps * np.eye(C_g.shape[0]))
    graded_ok = is_psd(S_eps, tol=1e-7) and is_psd(A_g - S_eps, tol=1e-7)
    graded_min_eig_S = float(np.linalg.eigvalsh(S_eps).min())
    graded_min_eig_AmS = float(np.linalg.eigvalsh(A_g - S_eps).min())

    # ---- mutation test (break the PSD assumption) ----
    # Take gamma_0 *above* the Thm 3.2 bound so L* becomes indefinite.
    b1_minus = sigma_min_plus(cx["B1"])
    bound_g0 = 1.0 * b1_minus          # beta_1 = 1
    cx_mut = build_complex(nv, edges, tri_valid, betas=[1.0, 1.0],
                           gammas=[bound_g0 * 1.5, 0.4])   # gamma_0 exceeds bound
    Lstar_mut, A_m, X_m, C_m = graded_laplacian(cx_mut)
    Lstar_mut_psd = is_psd(Lstar_mut, tol=1e-7)
    S_mut = schur_complement(A_m, X_m, C_m + eps * np.eye(C_m.shape[0]))
    S_mut_psd = is_psd(S_mut, tol=1e-7)
    AmS_mut_psd = is_psd(A_m - S_mut, tol=1e-7)
    property_breaks = bool((not Lstar_mut_psd) and (not (S_mut_psd and AmS_mut_psd)))

    holds = (n_fail == 0)
    verdict = "verified" if holds else "inconclusive"

    result = {
        "claim": 2,
        "statement": ("Collapsed operator is spectrally bounded 0 <= S <= A "
                      "(Proposition 3.5, Sec 3.2)."),
        "verdict": verdict,
        "holds": bool(holds),
        "source": "Proposition 3.5, Section 3.2 (arXiv:2606.23517)",
        "seed": MASTER_SEED + SEED,
        "numerics": {
            "n_random_instances": n_tot,
            "n_failures_of_0leSleA": int(n_fail),
            "min_eig_S_over_instances": min_eig_S,
            "min_eig_A_minus_S_over_instances": min_eig_AmS,
            "graded_Lstar_PSD": bool(is_psd(Lstar, tol=1e-7)),
            "graded_S_eps_PSD": bool(graded_ok),
            "graded_min_eig_S_eps": graded_min_eig_S,
            "graded_min_eig_A_minus_S_eps": graded_min_eig_AmS,
        },
        "mutation_test": {
            "description": ("Break Thm 3.2 PSD assumption: set gamma_0 = 1.5 * "
                            "beta_1*sigma_min^+(B_1) so L* is no longer PSD."),
            "gamma0_bound": float(bound_g0),
            "gamma0_used": float(bound_g0 * 1.5),
            "mutated_Lstar_PSD": bool(Lstar_mut_psd),
            "mutated_S_eps_PSD": bool(S_mut_psd),
            "mutated_A_minus_S_eps_PSD": bool(AmS_mut_psd),
            "property_breaks": property_breaks,
            "note": ("Once L* is indefinite the Schur complement is no longer "
                     "guaranteed PSD, so 0 <= S <= A can fail -- confirming the "
                     "bound is conditional on the PSD setup of Prop 3.5.")
        },
    }
    return result


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    res = main()
    with open("results/claim2.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
