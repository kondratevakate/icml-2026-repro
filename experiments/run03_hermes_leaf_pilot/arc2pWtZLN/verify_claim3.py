"""
Claim 3 (Corollary 3.6, Section 3.2):
  "Eigenvalue compression holds for every index k, with
   lambda_k(S) <= lambda_k(A), meaning the collapsed operator never exceeds
   the rank-0 Laplacian's spectrum."

Prerequisite: S <= A (Claim 2), i.e. A - S = X C^{-1} X^T >= 0.
For symmetric PSD matrices, S <= A implies lambda_k(S) <= lambda_k(A) for all k
(eigenvalue monotonicity / Weyl). Verification:
  * Over many seeded PSD block matrices (C PD) confirm lambda_k(S) <= lambda_k(A)
    for every k, and quantify the compression gap.
  * On the paper's graded-Laplacian collapse (regularized) confirm the same.

Mutation test:
  * Break S <= A by using the wrong-sign operator S_mut = A + X C^{-1} X^T.
    Then lambda_k(S_mut) can exceed lambda_k(A) -> the compression fails.
"""
import json
import numpy as np
from common import (rng, build_complex, graded_laplacian, schur_complement,
                    is_psd, MASTER_SEED)

SEED = 303


def make_psd_block(n, p, r):
    N = n + p
    R = r.standard_normal((N, N))
    M = R @ R.T + 0.1 * np.eye(N)
    return M[:n, :n], M[:n, n:], M[n:, n:]


def main():
    r = rng(SEED)
    n_viol = 0
    n_tot = 40
    max_gap = -np.inf          # lambda_k(A) - lambda_k(S), should be >=0
    min_gap = np.inf
    worst_k = -1
    for i in range(n_tot):
        ri = rng(SEED + i)
        n = 10 + (i % 4)
        p = 14 + (i % 5)
        A, X, C = make_psd_block(n, p, ri)
        S = schur_complement(A, X, C)
        wA = np.linalg.eigvalsh(A)
        wS = np.linalg.eigvalsh(S)
        gaps = wA - wS
        if np.min(gaps) < -1e-8:
            n_viol += 1
        g = float(np.min(gaps))
        if g > max_gap:
            max_gap = g
        if g < min_gap:
            min_gap = g
            worst_k = int(np.argmin(gaps))

    # ---- paper graded Laplacian collapse ----
    nv = 12
    edges = [(i, (i + 1) % nv) for i in range(nv)] + [(0, 6), (2, 9), (4, 10)]
    eset = set(edges) | {(b, a) for (a, b) in edges}
    raw_tris = [(0, 1, 6), (2, 3, 9), (4, 5, 10), (6, 7, 8), (9, 10, 11)]
    tri_valid = [t for t in raw_tris if all(((a, b) in eset or (b, a) in eset)
                   for a, b in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])]
    cx = build_complex(nv, edges, tri_valid, betas=[1.0, 1.0], gammas=[0.4, 0.4])
    Lstar, A_g, X_g, C_g = graded_laplacian(cx)
    eps = 1e-4
    S_g = schur_complement(A_g, X_g, C_g + eps * np.eye(C_g.shape[0]))
    wAg = np.linalg.eigvalsh(A_g)
    wSg = np.linalg.eigvalsh(S_g)
    graded_gaps = wAg - wSg
    graded_ok = bool(np.all(graded_gaps >= -1e-7))
    graded_min_gap = float(np.min(graded_gaps))
    graded_max_gap = float(np.max(graded_gaps))

    # ---- mutation test ----
    A_m, X_m, C_m = make_psd_block(12, 16, rng(SEED + 999))
    Cinv = np.linalg.inv(C_m)
    S_mut = A_m + X_m @ Cinv @ X_m.T          # wrong sign -> S_mut >= A, not <=
    wAm = np.linalg.eigvalsh(A_m)
    wSm = np.linalg.eigvalsh(S_mut)
    mut_violations = int(np.sum(wSm > wAm + 1e-8))

    holds = (n_viol == 0)
    result = {
        "claim": 3,
        "statement": ("Eigenvalue compression lambda_k(S) <= lambda_k(A) for every "
                      "k (Corollary 3.6, Sec 3.2)."),
        "verdict": "verified" if holds else "falsified",
        "holds": bool(holds),
        "source": "Corollary 3.6, Section 3.2 (arXiv:2606.23517)",
        "seed": MASTER_SEED + SEED,
        "numerics": {
            "n_random_instances": n_tot,
            "n_eigenvalue_violations": int(n_viol),
            "min_gap_lambda_k(A)-lambda_k(S)": float(min_gap),
            "max_gap_lambda_k(A)-lambda_k(S)": float(max_gap),
            "graded_min_gap": graded_min_gap,
            "graded_max_gap": graded_max_gap,
            "graded_all_compressed": graded_ok,
        },
        "mutation_test": {
            "description": "Use S_mut = A + X C^{-1} X^T (wrong sign, breaking S <= A).",
            "n_eigenvalue_violations_mutated": mut_violations,
            "property_breaks": bool(mut_violations > 0),
            "note": ("With the wrong sign A - S_mut = -2 X C^{-1} X^T <= 0, so "
                     "S_mut >= A and lambda_k(S_mut) >= lambda_k(A): the compression "
                     "reverses -> Claim 3 fails when S <= A is violated.")
        },
    }
    return result


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    res = main()
    with open("results/claim3.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
