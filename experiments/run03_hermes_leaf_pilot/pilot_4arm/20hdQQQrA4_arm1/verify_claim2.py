"""verify_claim2.py — Eq. (8) trainable null-space component w_phi(x).

Three executable sub-tests:
 A. Consistency: A_gamma P_gamma = b_gamma for EVERY w_phi (Eq. 16 of App. A).
 B. Degeneracy law: the output depends on w_phi iff ||I - A_g^+ A_g|| > 0, i.e. iff
    A_gamma is NOT full column rank (paper, Sec. 3.2). Exhaustive over ranks.
 C. Trainability / joint optimisation: on a task whose optimum lies on an edge of the
    feasible face, fitting w_phi (least squares over the null space) strictly beats the
    fixed orthogonal projection w_phi = 0.
 MUTATION for C: freeze w_phi = 0 (fixed orthogonal projection) -> loss must be >= trained.
"""
import json, time
import numpy as np
from caffnet_core import P_gamma, caffnet, gamma_set, random_feasible_instance

OUT = "results/claim2.json"
rng_global = np.random.default_rng(0)


def test_A(n_seeds=200):
    worst = 0.0
    for s in range(n_seeds):
        rng = np.random.default_rng(1000 + s)
        n_out = int(rng.integers(1, 6)); m = int(rng.integers(1, 8))
        rank = int(rng.integers(1, min(m, n_out) + 1))
        A, b, y0 = random_feasible_instance(rng, n_out, m, rank=rank, redundant=(s % 2 == 0))
        f = rng.normal(size=n_out)
        for gam in gamma_set(m, n_out):
            idx = list(gam); Ag, bg = A[idx, :], b[idx]
            if np.linalg.norm(Ag @ np.linalg.pinv(Ag) @ bg - bg) > 1e-9:
                continue                       # inconsistent sub-system, Eq.16 n/a
            for _ in range(5):
                w = rng.normal(size=n_out) * 10
                worst = max(worst, float(np.max(np.abs(Ag @ P_gamma(f, A, b, gam, w) - bg))))
    return dict(max_abs_residual_Ag_Pg_minus_bg=worst, n_seeds=n_seeds)


def test_B(n_seeds=200):
    full_col_rank_spread, deficient_spread = [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(2000 + s)
        n_out = int(rng.integers(1, 5)); m = int(rng.integers(1, 7))
        rank = int(rng.integers(1, min(m, n_out) + 1))
        A, b, y0 = random_feasible_instance(rng, n_out, m, rank=rank)
        f = rng.normal(size=n_out)
        for gam in gamma_set(m, n_out):
            Ag = A[list(gam), :]
            Ap = np.linalg.pinv(Ag)
            N = np.eye(n_out) - Ap @ Ag
            ys = [P_gamma(f, A, b, gam, rng.normal(size=n_out)) for _ in range(8)]
            spread = float(np.max(np.std(np.array(ys), axis=0)))
            (full_col_rank_spread if np.linalg.norm(N) < 1e-9
             else deficient_spread).append(spread)
    return dict(
        n_full_col_rank_cases=len(full_col_rank_spread),
        max_spread_when_full_col_rank=float(max(full_col_rank_spread, default=0.0)),
        n_rank_deficient_cases=len(deficient_spread),
        min_spread_when_rank_deficient=float(min(deficient_spread, default=0.0)),
        median_spread_when_rank_deficient=float(np.median(deficient_spread) if deficient_spread else 0.0))


def _loss(W, X, Y, As, bs, p=2):
    tot = 0.0
    for i in range(len(X)):
        f = np.array([X[i], -X[i]])          # fixed "unconstrained net" output
        y = caffnet(f, As[i], bs[i], W, p=p)
        tot += float(np.sum((y - Y[i]) ** 2))
    return tot / len(X)


def test_C(n_seeds=8):
    """Task: n_out=2, one active constraint (rank 1 -> 1-D null space). The target sits at a
    specific point ON the constraint line, not at the orthogonal projection."""
    trained, orthogonal = [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(3000 + s)
        X = rng.uniform(-1, 1, size=40)
        As, bs, Y = [], [], []
        a = rng.normal(size=2); a /= np.linalg.norm(a)
        t = np.array([-a[1], a[0]])          # direction of the line
        for x in X:
            A = np.vstack([a, 2 * a])        # linearly dependent, rank 1, m=2 > rank
            b = np.array([-1.0, -2.0])       # a^T y <= -1 (violated by f)
            y_star = a * (-1.0) + 0.7 * t    # target on the line, offset along null space
            As.append(A); bs.append(b); Y.append(y_star)
        # grid + refine search over the scalar null-space coefficient
        cand = np.linspace(-3, 3, 121)
        losses = [_loss(c * t, X, Y, As, bs) for c in cand]
        c0 = cand[int(np.argmin(losses))]
        fine = np.linspace(c0 - 0.05, c0 + 0.05, 41)
        lf = [_loss(c * t, X, Y, As, bs) for c in fine]
        trained.append(float(min(lf)))
        orthogonal.append(float(_loss(np.zeros(2), X, Y, As, bs)))   # MUTATION: w_phi = 0
    return dict(n_seeds=n_seeds,
                loss_trained_w=trained, loss_w_zero_orthogonal=orthogonal,
                mean_loss_trained=float(np.mean(trained)),
                mean_loss_orthogonal=float(np.mean(orthogonal)),
                seeds_where_trained_strictly_better=int(sum(
                    1 for a, b in zip(trained, orthogonal) if a < b - 1e-12)))


if __name__ == "__main__":
    t0 = time.time()
    res = dict(claim=2, A_consistency=test_A(), B_degeneracy=test_B(), C_trainability=test_C(),
               command="python verify_claim2.py", numpy=np.__version__,
               seconds=round(time.time() - t0, 2))
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))
