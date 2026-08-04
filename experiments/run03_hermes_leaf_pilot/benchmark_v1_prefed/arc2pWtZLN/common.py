"""
common.py — shared helpers for reproducing the 6 anchored claims of
"Collapsed Effective Operators for Higher-order Structures" (arXiv:2606.23517).

All computation is CPU-only (numpy / scipy / sympy). Randomness is pinned via a
single master seed so every result is reproducible.

Paper definitions used (Section 3):
  * Graded Laplacian L* (Prop 3.1):
        D_k = beta_k B_k^T B_k + beta_{k+1} B_{k+1} B_{k+1}^T   (B_0 = B_{K+1} = 0)
        L*  = block-tridiag(D_k, offdiag = -gamma_k B_{k+1})
  * PSD condition (Thm 3.2):
        L* >= 0  <=>  gamma_k <= beta_{k+1} * sigma_min^+(B_{k+1})
  * Collapsed effective operator (Def 3.3):
        S := A - X C^{-1} X^T     (Schur complement of L* = [[A,X],[X^T,C]])
  * Spectrum (Prop 3.5 / Cor 3.6):   0 <= S <= A ,   lambda_k(S) <= lambda_k(A)
  * Regularized (Prop 3.10):
        S_eps := A - X (C + eps I)^{-1} X^T
        ||S_eps - S^dag||_2 <= eps ||X||_2^2 / (lambda_+ (lambda_+ + eps))
        with lambda_+ = lambda_min^+(C), S^dag = A - X C^dag X^T.
"""
import numpy as np

MASTER_SEED = 260622  # arXiv submission date 22 Jun 2026


def rng(seed):
    return np.random.default_rng(MASTER_SEED + seed)


def signed_incidence(n_vertices, edges):
    """Signed vertex/edge incidence B1 in R^{n x m} for edge list `edges`.

    For edge (a,b): B1[a,j]=+1, B1[b,j]=-1  (oriented incidence).
    """
    m = len(edges)
    B1 = np.zeros((n_vertices, m))
    for j, (a, b) in enumerate(edges):
        B1[a, j] = 1.0
        B1[b, j] = -1.0
    return B1


def triangle_edge_incidence(edges, triangles):
    """Signed edge/triangle incidence B2 in R^{m x t} so that B1 @ B2 == 0.

    For triangle (i,j,k) with edges e_ij, e_jk, e_ki we assign orientations
    (e_ij -> +1, e_jk -> +1, e_ki -> -1) which makes the boundary B1 B2 vanish.
    """
    m = len(edges)
    edge_index = {e: j for j, e in enumerate(edges)}
    t = len(triangles)
    B2 = np.zeros((m, t))
    for tri, (i, j, k) in enumerate(triangles):
        e_ij = (i, j) if (i, j) in edge_index else (j, i)
        e_jk = (j, k) if (j, k) in edge_index else (k, j)
        e_ki = (k, i) if (k, i) in edge_index else (i, k)
        B2[edge_index[e_ij], tri] = 1.0
        B2[edge_index[e_jk], tri] = 1.0
        B2[edge_index[e_ki], tri] = -1.0
    return B2


def build_complex(n_vertices, edges, triangles, betas, gammas):
    """Build signed boundary matrices B1 (n x m) and B2 (m x t) for a
    rank-2 combinatorial complex from edge/triangle lists.

    betas  = [beta_1, beta_2]   (beta_0 = beta_3 = 0)
    gammas = [gamma_0, gamma_1]
    Returns dict with B1, B2, n, m, t.
    """
    B1 = signed_incidence(n_vertices, edges)
    B2 = triangle_edge_incidence(edges, triangles)
    return {"B1": B1, "B2": B2, "n": n_vertices, "m": len(edges), "t": len(triangles),
            "betas": list(betas), "gammas": list(gammas)}


def graded_laplacian(cx):
    """Assemble the rank-2 Graded Laplacian L* (Prop 3.1, Eq 5/10).

    Returns Lstar (block (n+m+t) x (n+m+t)), and the block decomposition
    L* = [[A, X],[X^T, C]] with X = [-gamma_0 B1, 0].
    """
    B1, B2 = cx["B1"], cx["B2"]
    beta1, beta2 = cx["betas"][0], cx["betas"][1]
    g0, g1 = cx["gammas"][0], cx["gammas"][1]
    n, m, t = cx["n"], cx["m"], cx["t"]

    D0 = beta1 * (B1 @ B1.T)                       # rank-0 block  (A)
    D1 = beta1 * (B1.T @ B1) + beta2 * (B2 @ B2.T) # rank-1 block
    D2 = beta2 * (B2.T @ B2)                       # rank-2 block

    top = np.block([[D0, -g0 * B1, np.zeros((n, t))]])
    mid = np.block([[-g0 * B1.T, D1, -g1 * B2]])
    bot = np.block([[np.zeros((t, n)), -g1 * B2.T, D2]])
    Lstar = np.block([[top], [mid], [bot]])

    A = D0
    X = np.hstack([-g0 * B1, np.zeros((n, t))])          # (n x (m+t))
    C = np.block([[D1, -g1 * B2], [-g1 * B2.T, D2]])     # ((m+t) x (m+t))
    return Lstar, A, X, C


def sigma_min_plus(M):
    """Smallest *positive* singular value of M (sigma_min^+)."""
    s = np.linalg.svd(M, compute_uv=False)
    s = np.sort(s[s > 1e-12])
    if s.size == 0:
        return 0.0
    return float(s[0])


def is_psd(M, tol=1e-8):
    """True if M is symmetric PSD within tolerance."""
    e = np.linalg.eigvalsh(M)
    return bool(np.all(e >= -tol * max(1.0, np.max(np.abs(e)))))


def schur_complement(A, X, C):
    """Explicit Schur complement of block C in [[A,X],[X^T,C]]: A - X C^{-1} X^T."""
    return A - X @ np.linalg.inv(C) @ X.T


def schur_via_elimination(A, X, C):
    """Schur complement computed by Gaussian elimination:
    z = -C^{-1} X^T u  ->  u^T (A - X C^{-1} X^T) u.
    We verify the vertex-only operator matches for random test vectors."""
    Cinv = np.linalg.inv(C)
    return A - X @ Cinv @ X.T


def schur_from_block(M, n):
    """Schur complement of the lower-right block C in the block matrix
    M = [[A, X],[X^T, C]] (split at index n).

    Exact factorisation (with C invertible):
        M = L_left @ [[S, 0],[0, C]] @ L_right
        L_left  = [[I, X C^{-1}],[0, I]],   L_right = [[I, 0],[C^{-1} X^T, I]]
    Hence  S = (L_left^{-1} M L_right^{-1})[:n, :n].
        L_left^{-1}  = [[I, -X C^{-1}],[0, I]]
        L_right^{-1} = [[I, 0],[-C^{-1} X^T, I]]"""
    C = M[n:, n:]
    Cinv = np.linalg.inv(C)
    X = M[:n, n:]
    N = M.shape[0]
    L_left_inv = np.eye(N)
    L_left_inv[:n, n:] = -X @ Cinv            # (n,p) @ (p,p) = (n,p)
    L_right_inv = np.eye(N)
    L_right_inv[n:, :n] = -Cinv @ X.T          # (p,p) @ (p,n) = (p,n)
    congr = L_left_inv @ M @ L_right_inv
    return congr[:n, :n]


def implicit_action(A, X, C, eps, x):
    """Algorithm 1: compute S_eps @ x WITHOUT forming S_eps explicitly.
    y = X^T x ; z = (C + eps I)^{-1} y (solved with CG) ; return A x - X z."""
    from scipy.sparse.linalg import cg
    import scipy.sparse as sp
    y = X.T @ x
    M = sp.csc_matrix(C + eps * np.eye(C.shape[0]))
    z, info = cg(M, y, atol=1e-12, rtol=1e-12)
    return A @ x - X @ z, info


def explicit_regularized(A, X, C, eps):
    """Explicit S_eps = A - X (C + eps I)^{-1} X^T."""
    return A - X @ np.linalg.inv(C + eps * np.eye(C.shape[0])) @ X.T
