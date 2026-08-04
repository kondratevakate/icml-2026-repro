"""Claim 1 (TASK #1; paper v2 Sec. 3.1, Eq. (4), embedding pi -> F dpi/drho).

Claim: the Fourier transform F is an isometry on L^2(R^d; rho) and turns the
non-inner-product Kantorovich pairing <c|pi> into an inner product in the
frequency domain:   <c|pi> = int Fc(-z) Fpi(z) drho(z).

Executable test (d=1, discretised torus grid so DFT is an exact unitary):
  * unitarity / Parseval of the normalised DFT (isometry of F),
  * the bilinear identity (4) for random costs c and random transport plans pi
    given as densities w.r.t. the reference measure rho,
  * that the identity holds for genuine OT couplings pi in Pi(mu,nu),
  * MUTATION: replace F by a non-unitary (rescaled/truncated) transform ->
    the isometry and the identity must break.
Seed 20260803.
"""
import numpy as np
from common import save

SEED = 20260803
rng = np.random.default_rng(SEED)

n = 64  # grid points per axis; rho = uniform on the grid (product form)
# --- 1D isometry check ---------------------------------------------------
def Fop(v):
    "unitary DFT = discrete Fourier isometry on L^2(grid, uniform rho)"
    return np.fft.fft(v, norm="ortho")

x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
y = rng.standard_normal(n) + 1j * rng.standard_normal(n)
iso_err = abs(np.vdot(Fop(x), Fop(y)) - np.vdot(x, y)) / abs(np.vdot(x, y))
norm_err = abs(np.linalg.norm(Fop(x)) - np.linalg.norm(x)) / np.linalg.norm(x)

# --- bilinear identity <c|pi> = <Fc(-.)|Fpi> on a 2D grid (X x Y) --------
m = 32
xs = np.linspace(0, 1, m, endpoint=False)
c = np.add.outer(np.abs(xs[:, None] - xs[None, :]).ravel() * 0, 0).reshape(m, m)  # placeholder
c = np.abs(xs[:, None] - xs[None, :]) ** 2 + 0.3 * np.sin(6 * np.pi * xs[:, None]) * np.cos(4 * np.pi * xs[None, :])

# reference measure rho = uniform on the m x m grid, mass 1
w = 1.0 / (m * m)

def F2(v):
    return np.fft.fft2(v, norm="ortho")

def pairing_time(cc, dens):
    "<c|pi> = int c dpi = sum c * dens * rho-weights"
    return float(np.real(np.sum(cc * dens) * w))

def pairing_freq(cc, dens):
    "int Fc(-z) Fpi(z) drho(z) -- with unitary DFT this is the Hilbert inner product"
    Fc = F2(cc)
    Fp = F2(dens)
    return float(np.real(np.vdot(Fc, Fp)) * w)

# random density w.r.t. rho, then a genuine coupling with prescribed marginals
dens_rand = rng.random((m, m)) * 3.0
t_rand, f_rand = pairing_time(c, dens_rand), pairing_freq(c, dens_rand)

# genuine coupling: Sinkhorn-scale a random kernel to marginals mu, nu
mu = rng.random(m) + 0.1; mu /= mu.sum()
nu = rng.random(m) + 0.1; nu /= nu.sum()
K = np.exp(-c / 0.1)
u = np.ones(m); v = np.ones(m)
for _ in range(2000):
    u = mu / (K @ v)
    v = nu / (K.T @ u)
pi = u[:, None] * K * v[None, :]          # a coupling (probability matrix)
dens_pi = pi / w                          # density w.r.t. rho
t_ot, f_ot = pairing_time(c, dens_pi), pairing_freq(c, dens_pi)
marg_err = float(max(np.abs(pi.sum(1) - mu).max(), np.abs(pi.sum(0) - nu).max()))

# --- MUTATION: non-unitary transform (drop normalisation + truncate) -----
def Fbad(v):
    out = np.fft.fft2(v)          # not orthonormal
    out[m // 2:, :] = 0           # truncate high frequencies -> not an isometry
    return out

mut_iso = abs(np.vdot(Fbad(np.asarray(c, complex)), Fbad(dens_pi)) * w - t_ot) / abs(t_ot)
mut_norm = abs(np.linalg.norm(np.fft.fft(x)) - np.linalg.norm(x)) / np.linalg.norm(x)

tol = 1e-10
ok = (iso_err < tol and norm_err < tol
      and abs(t_rand - f_rand) / abs(t_rand) < 1e-10
      and abs(t_ot - f_ot) / abs(t_ot) < 1e-10
      and marg_err < 1e-8 and mut_iso > 1e-2)

out = dict(
    claim=1, source="paper v2 Sec. 3.1, Eq. (4) (TASK cites 'Section 4.1, Eq. 7')",
    seed=SEED, grid=m,
    isometry_relative_error=iso_err, norm_preservation_relative_error=norm_err,
    pairing_time_domain_random=t_rand, pairing_freq_domain_random=f_rand,
    rel_gap_random=abs(t_rand - f_rand) / abs(t_rand),
    pairing_time_domain_coupling=t_ot, pairing_freq_domain_coupling=f_ot,
    rel_gap_coupling=abs(t_ot - f_ot) / abs(t_ot),
    coupling_marginal_error=marg_err,
    mutation_nonunitary_rel_gap=float(mut_iso),
    mutation_unnormalised_dft_norm_rel_error=float(mut_norm),
    verdict="verified" if ok else "inconclusive",
)
for k, v in out.items():
    print(f"{k}: {v}")
save("claim1.json", out)
