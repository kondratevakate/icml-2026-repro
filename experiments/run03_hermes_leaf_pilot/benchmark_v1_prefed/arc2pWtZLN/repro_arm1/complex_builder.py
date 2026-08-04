"""
Shared builder for a concrete combinatorial/simplicial complex used to
reproduce the 6 anchored claims of "Collapsed Effective Operators for
Higher-order Structures" (arXiv:2606.23517 / OpenReview arc2pWtZLN).

We use a simplicial complex (a concrete instance of a Combinatorial
Complex, Def. 3.9) so the boundary matrices B1, B2 are the standard
signed cellular boundary matrices, exactly matching Example 3.4.

Complex chosen: a wheel/disk on 5 vertices (center 4 + rim 0,1,2,3),
8 edges, 4 triangular faces.  This complex has:
  - beta2 = 0  (no 2D hole)  -> B2 has full column rank (good for C PD)
  - the 1-cycle space (dim n1 - n0 + 1 = 4) is spanned by the 4 faces,
    so B2 B2^T fills the kernel of B1^T B1 and the higher-order block C
    is (experimentally) positive definite -> claims 1-3 can use exact C^-1.

All functions are pure (deterministic) given the module-level complex.
Seeds are pinned inside the verify_*.py scripts, not here.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Concrete complex definition (simplicial complex = CC with rank adjacency)
# ---------------------------------------------------------------------------
# Rank-0 cells: vertices
VERTICES = [0, 1, 2, 3, 4]          # n0 = 5
# Rank-1 cells: edges  (sorted vertex tuples)
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # rim
    (0, 4), (1, 4), (2, 4), (3, 4),  # spokes
]                                     # n1 = 8
# Rank-2 cells: triangular faces  (sorted vertex tuples)
FACES = [
    (0, 1, 4),
    (1, 2, 4),
    (2, 3, 4),
    (3, 0, 4),
]                                     # n2 = 4


def _orient(simplex):
    """Return simplex vertices sorted ascending (canonical orientation)."""
    return tuple(sorted(simplex))


def build_B1():
    """Signed boundary matrix B1 in R^{n0 x n1}.

    B1[v, e] = +1 if v is the head of edge e, -1 if tail, 0 otherwise.
    Edge e = (a, b) with a < b: head = b, tail = a.
    """
    n0 = len(VERTICES)
    n1 = len(EDGES)
    B1 = np.zeros((n0, n1))
    vidx = {v: i for i, v in enumerate(VERTICES)}
    for j, e in enumerate(EDGES):
        a, b = e  # a < b by construction
        B1[vidx[a], j] = -1.0
        B1[vidx[b], j] = +1.0
    return B1


def build_B2():
    """Signed boundary matrix B2 in R^{n1 x n2}.

    For a face f = (v0 < v1 < v2), the (k-1)-faces are the 3 edges obtained
    by deleting vertex at index i, with sign (-1)^i.
    """
    n1 = len(EDGES)
    n2 = len(FACES)
    B2 = np.zeros((n1, n2))
    eidx = {tuple(sorted(e)): j for j, e in enumerate(EDGES)}
    for k, f in enumerate(FACES):
        f = tuple(sorted(f))
        # three (k-1)-faces = edges of the triangle
        sub = [
            (f[1], f[2]),  # delete f[0] -> sign (-1)^0 = +1
            (f[0], f[2]),  # delete f[1] -> sign (-1)^1 = -1
            (f[0], f[1]),  # delete f[2] -> sign (-1)^2 = +1
        ]
        signs = [+1.0, -1.0, +1.0]
        for (edge, sgn) in zip(sub, signs):
            edge = tuple(sorted(edge))
            j = eidx[edge]
            B2[j, k] += sgn
    return B2


def build_blocks(betas, gammas, ridge_eps=0.0):
    """Build the block decomposition of the graded Laplacian (Eq. 5 / 8).

    Parameters
    ----------
    betas   : dict-like with keys 1 (=beta1) and 2 (=beta2)  (beta0,beta3 unused; 0)
    gammas  : dict-like with keys 0 (=gamma0) and 1 (=gamma1)
    ridge_eps : tiny diagonal load added to the higher-order block C so that
                C is strictly positive definite (mirrors the paper's Tikhonov
                regularization of Sec. 3.4). 0.0 means use exact C.

    Returns
    -------
    A, X, C, Lstar  (numpy arrays)
        Lstar = [[A, X], [X^T, C]]   (block form Eq. 8)
        A      : n0 x n0  vertex (rank-0) block
        C      : (n1+n2) x (n1+n2)  higher-order block
        X      : n0 x (n1+n2)  coupling block  ( = [-gamma0 B1 | 0] )
    """
    B1 = build_B1()
    B2 = build_B2()
    b1 = float(betas[1])
    b2 = float(betas[2])
    g0 = float(gammas[0])
    g1 = float(gammas[1])

    n0 = B1.shape[0]
    n1 = B1.shape[1]
    n2 = B2.shape[1]

    # Diagonal blocks (Eq. 6):  D^k = beta_k B_k^T B_k + beta_{k+1} B_{k+1} B_{k+1}^T
    # B0 = B3 = 0.
    D0 = b1 * (B1 @ B1.T)                       # n0 x n0  (vertex block)
    D1 = b1 * (B1.T @ B1) + b2 * (B2 @ B2.T)    # n1 x n1
    D2 = b2 * (B2.T @ B2)                       # n2 x n2

    # Block partition (Eq. 8):  Lstar = [[A, X],[X^T, C]]
    A = D0                                            # n0 x n0
    X = np.block([-g0 * B1, np.zeros((n0, n2))])      # n0 x (n1+n2)
    C = np.block([
        [D1,              -g1 * B2],
        [-g1 * B2.T,       D2],
    ])                                                # (n1+n2) x (n1+n2)
    if ridge_eps > 0.0:
        C = C + ridge_eps * np.eye(C.shape[0])
    # Assemble Lstar directly from the block partition so any ridge on C is
    # reflected consistently in the bottom-right block (Eq. 8).
    Lstar = np.block([[A, X], [X.T, C]])

    return A, X, C, Lstar


def sigma_min_plus(M):
    """Smallest POSITIVE singular value of M (sigma_min^+ in Thm 3.2)."""
    s = np.linalg.svd(M, compute_uv=False)
    pos = s[s > 1e-12]
    if pos.size == 0:
        return 0.0
    return float(pos.min())


def psd_min_eig(M):
    """Smallest eigenvalue of a symmetric matrix (numerical)."""
    w = np.linalg.eigvalsh(M)
    return float(w.min()), w


if __name__ == "__main__":
    B1 = build_B1()
    B2 = build_B2()
    print("n0,n1,n2 =", B1.shape[0], B1.shape[1], B2.shape[1])
    print("rank B1 =", np.linalg.matrix_rank(B1), " sigma_min^+ =", sigma_min_plus(B1))
    print("rank B2 =", np.linalg.matrix_rank(B2), " sigma_min^+ =", sigma_min_plus(B2))
    A, X, C, L = build_blocks({1: 1.0, 2: 1.0}, {0: 1.0, 1: 1.0}, ridge_eps=0.0)
    print("rank C =", np.linalg.matrix_rank(C), "of", C.shape[0],
          " min eig C =", np.linalg.eigvalsh(C).min())
    print("rank Lstar =", np.linalg.matrix_rank(L), "of", L.shape[0])
