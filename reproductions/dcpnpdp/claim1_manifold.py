"""
DC-PnPDP (OpenReview jEBkuuETjr / arXiv 2602.23214) -- Claim 1 reproduction, CPU only.

CLAIM 1 (marketing sentence, paper intro):
  "Theoretically guarantees asymptotic convergence to exact data manifold through
   dual variable coupling."

TRANSCRIBED SPEC (from arXiv HTML 2602.23214v1, Appendices C/D and Algorithm 1)
------------------------------------------------------------------------------
Assumption D.1 (Denoiser-Proximal Equivalence):
  "the pretrained denoiser D_sigma acts locally as the proximal operator of the
   regularizer phi with parameter gamma > 0", i.e.
     D_sigma(v) ~= prox_{gamma phi}(v) = argmin_z 0.5||z-v||^2 + gamma phi(z).

Theorem C.1 / D.2 (Fixed-point optimality):
  "Let (x*,z*,u*) be a fixed point of the deterministic backbone of the proposed
   Dual-Coupled PnP ADMM iterations. Under Assumption D.1, x* satisfies the
   first-order optimality condition of the original problem:
       0 in grad f(x*) + d phi(x*)
   with effective regularization weight lambda = rho*gamma."
  Contrast (loose coupling, u discarded):
       0 in grad f(x~) + d phi(x~) + (x~ - D_sigma(x~))     <- systematic bias.
  Paper's own caveat: "establishing global convergence guarantees for non-convex
  PnP algorithms with stochastic diffusion priors remains an open theoretical
  challenge."

Algorithm 1 (deterministic backbone used here; SH is a noise-injection step that
is inert when the denoiser is exact, so the backbone is the object of Claim 1):
  x^{k+1} = argmin_x ||Ax-y||^2 + lambda||x - z^k + u^k||^2
  v^{k+1} = x^{k+1} + u^k          (+ SpectralHomogenize)
  z^{k+1} = D_theta(v~^{k+1}, t)
  u^{k+1} = u^k + (x^{k+1} - z^{k+1})

WHAT "CONVERGENCE TO THE EXACT DATA MANIFOLD" WOULD MEAN
  dist(x^k, M) -> 0, where M = {x : phi(x) = 0} is the clean data manifold.

TEST DESIGN
  M = a known r-dim linear subspace of R^n (columns of orthonormal U).
  phi(x) = (1/(2 gamma)) * dist(x, M)^2  =>  prox_{gamma phi} is EXACTLY the
  shrink-toward-subspace map  v -> P v + (1-a)(v - P v)... concretely
  prox(v) = P v + (1/(1+1)) (I-P) v  for the chosen gamma (see code), a bona fide
  proximal denoiser satisfying Assumption D.1 EXACTLY (non-vacuous, best case).
  Limit s -> 0 of the shrink factor = hard projection onto M (idealized denoiser).
  y = A x_true + noise, x_true in M.
  Baseline: identical iteration with u frozen at 0 ("loose coupling"/memoryless PnP).
  Metric: dist(x^k, M)/||x^k|| at convergence, over 12 seeds.
"""
import numpy as np

N, R, M_MEAS, ITERS = 60, 8, 40, 4000
LAM = 1.0


def prox_shrink(v, P, s):
    """prox of phi(x)= (1/(2 gamma))dist(x,M)^2 -> keeps P v, shrinks normal part by s."""
    pv = P @ v
    return pv + s * (v - pv)


def run(A, y, P, s, dual, lam=LAM, iters=ITERS):
    n = A.shape[1]
    AtA = A.T @ A
    Aty = A.T @ y
    Minv = np.linalg.inv(AtA + lam * np.eye(n))
    x = np.zeros(n); z = np.zeros(n); u = np.zeros(n)
    for _ in range(iters):
        x = Minv @ (Aty + lam * (z - u))
        z = prox_shrink(x + u, P, s)
        if dual:
            u = u + (x - z)
    return x, z


def dist_to_M(x, P):
    return np.linalg.norm(x - P @ x) / (np.linalg.norm(x) + 1e-12)


def main():
    rows = []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        U, _ = np.linalg.qr(rng.standard_normal((N, R)))
        P = U @ U.T
        A = rng.standard_normal((M_MEAS, N)) / np.sqrt(M_MEAS)
        x_true = U @ rng.standard_normal(R)
        y = A @ x_true + 0.01 * rng.standard_normal(M_MEAS)
        out = {"seed": seed}
        for s in (0.5, 0.1, 0.01, 0.0):
            xd, _ = run(A, y, P, s, dual=True)
            xn, _ = run(A, y, P, s, dual=False)
            out[s] = (dist_to_M(xd, P), dist_to_M(xn, P),
                      np.linalg.norm(xd - x_true) / np.linalg.norm(x_true))
        rows.append(out)

    print(f"{'seed':>4} " + " ".join(
        f"{'s=%g dual/none' % s:>26}" for s in (0.5, 0.1, 0.01, 0.0)))
    for r in rows:
        line = f"{r['seed']:>4} "
        for s in (0.5, 0.1, 0.01, 0.0):
            d, nn, _ = r[s]
            line += f"{d:11.2e}/{nn:<14.2e}"
        print(line)
    print("\nmedians over 12 seeds:")
    for s in (0.5, 0.1, 0.01, 0.0):
        d = np.median([r[s][0] for r in rows])
        nn = np.median([r[s][1] for r in rows])
        e = np.median([r[s][2] for r in rows])
        print(f"  s={s:<5} dist(x,M) dual={d:.3e}  loose={nn:.3e}   rel.err(dual)={e:.3e}")


if __name__ == "__main__":
    main()
