"""
Reproduction (CPU, classical single-point path) of the COVERAGE GUARANTEE of:
  "Principled Confidence Estimation for Deep Computed Tomography"
  Gaetzner & Kirschner, ICML 2026 (orid a6vCNSBBeq, arXiv 2602.05812).

We reproduce Proposition 3.2 / Theorem 3.1 (Sequential Likelihood Mixing) on a synthetic
phantom -- no LIDC data, no trained checkpoints. This covers the CLASSICAL half of the paper:
that the SLM confidence sequence attains its nominal (1-delta) coverage of the true image x*.

Forward model (Beer-Lambert, Poisson): for view angle alpha, R_alpha x = parallel-beam line
integrals; mean detector counts lambda = I0 * exp(-R_alpha x); measurement y ~ Poisson(lambda).

SLM confidence sequence (Prop 3.2, single-point mixing mu_{s-1}=delta_{xhat_{s-1}}):
  nll(x, s) = -sum_detectors [ y_s * log(lambda) - lambda ],   lambda = I0 exp(-R_{alpha_s} x)
  L_t(x*) = sum_{s<=t} nll(x*, s)        (data NLL at the truth)
  beta_t  = sum_{s<=t} nll(xhat_{s-1}, s)  (data NLL at the predictable reconstruction)
  C_t = { x : L_t(x) <= beta_t + log(1/delta) }.  Coverage: P(for all t: x* in C_t) >= 1-delta.
  The Poisson normalisation log(y!) is identical in L_t and beta_t and CANCELS in membership.

Test: over many independent Poisson realisations, empirical P(max_t [L_t(x*)-beta_t] <= log(1/delta))
must be >= 1-delta. xhat_{s-1} is reconstructed from views 1..s-1 only (predictable).
"""
import numpy as np

# --------------------------- parallel-beam Radon (bilinear rotation) --------------------
def _rotate(img, angle_deg):
    r = img.shape[0]
    c = (r - 1) / 2.0
    th = np.deg2rad(angle_deg)
    cs, sn = np.cos(th), np.sin(th)
    ys, xs = np.mgrid[0:r, 0:r].astype(np.float64)
    xr, yr = xs - c, ys - c
    sx = cs * xr + sn * yr + c          # inverse-rotated source coords
    sy = -sn * xr + cs * yr + c
    x0 = np.floor(sx).astype(int); y0 = np.floor(sy).astype(int)
    wx, wy = sx - x0, sy - y0
    def g(yy, xx):
        m = (yy >= 0) & (yy < r) & (xx >= 0) & (xx < r)
        return img[np.clip(yy, 0, r - 1), np.clip(xx, 0, r - 1)] * m
    return (g(y0, x0) * (1 - wx) * (1 - wy) + g(y0, x0 + 1) * wx * (1 - wy)
            + g(y0 + 1, x0) * (1 - wx) * wy + g(y0 + 1, x0 + 1) * wx * wy)

def radon_view(x, angle):
    return _rotate(x, angle).sum(axis=0)          # r detector line-integrals

def backproject(p, angle, r):
    return _rotate(np.tile(p, (r, 1)), -angle)    # adjoint (spread + rotate back)


# --------------------------------- forward + likelihood ---------------------------------
def mean_counts(x, angle, I0):
    return I0 * np.exp(-radon_view(x, angle))

def nll_kernel(x, y, angle, I0):
    """-(y log lambda - lambda) summed over detectors (Poisson log(y!) dropped -> cancels)."""
    lam = np.maximum(mean_counts(x, angle, I0), 1e-12)
    return -np.sum(y * np.log(lam) - lam)


# ------------------------------------- phantom ------------------------------------------
def phantom(r=32):
    """Simple attenuation phantom: a few overlapping discs, values ~ O(0.1)."""
    yy, xx = np.mgrid[0:r, 0:r].astype(np.float64)
    c = (r - 1) / 2.0
    x = np.zeros((r, r))
    for (cy, cx, rad, val) in [(c, c, r*0.38, 0.10), (c-r*0.12, c, r*0.18, 0.06),
                               (c+r*0.15, c-r*0.12, r*0.10, 0.08)]:
        x += val * (((yy - cy) ** 2 + (xx - cx) ** 2) <= rad ** 2)
    return x


# ---------------------------- one experiment (coverage of x*) ---------------------------
def one_run(xstar, angles, I0, delta, recon_lr=0.02, seed=0):
    rng = np.random.default_rng(seed)
    r = xstar.shape[0]
    xhat = np.zeros((r, r))            # xhat_0 (predictable prior before any data)
    M = 0.0                            # running L_t(x*) - beta_t
    max_M = -np.inf
    for s, a in enumerate(angles):
        lam = mean_counts(xstar, a, I0)
        y = rng.poisson(lam).astype(np.float64)                 # measurement y_s
        # term_s uses xhat_{s-1} (predictable) for beta and x* for L
        M += nll_kernel(xstar, y, a, I0) - nll_kernel(xhat, y, a, I0)
        max_M = max(max_M, M)
        # AFTER using xhat_{s-1}, update reconstruction with view s (log-domain SART step)
        p_meas = -np.log(np.maximum(y, 0.5) / I0)               # measured line integral
        resid = p_meas - radon_view(xhat, a)
        xhat = np.clip(xhat + recon_lr * backproject(resid, a, r) / r, 0.0, None)
    return max_M <= np.log(1.0 / delta)                          # x* in C_t for all t?


def coverage_experiment(r=32, n_angles=40, I0=1e4, delta=0.1, n_runs=300):
    xstar = phantom(r)
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    covered = sum(one_run(xstar, angles, I0, delta, seed=k) for k in range(n_runs))
    emp = covered / n_runs
    print(f"phantom r={r}, views={n_angles}, I0={I0:.0e}, delta={delta}, runs={n_runs}")
    print(f"nominal coverage 1-delta = {1-delta:.3f}")
    print(f"empirical coverage       = {emp:.3f}   ({covered}/{n_runs})")
    print("PASS: coverage >= 1-delta" if emp >= 1 - delta else
          "CHECK: coverage below nominal")
    return emp


if __name__ == "__main__":
    import sys
    if "--smoke" in sys.argv:
        coverage_experiment(r=24, n_angles=20, n_runs=40)
    else:
        for d in (0.1, 0.05):
            coverage_experiment(delta=d)
            print()
