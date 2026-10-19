"""Schema-faithful SYNTHETIC data for the CycMit-MRI pilot (no real data is read or used).

Schema taken from the repository's loaders (DataLoader.py, Mitigation.py, Config.yaml) and the paper:
  * DataLoaderSL reads <name>.mat with keys "kspace" and "coils", each stored as (nc, ky, kx) and
    transposed to (nc, kx, ky) by the loader; complex64 (the network weights are float32).
  * Mitigation.py reads data/omega_mask.mat and data/delta{1,2,3}.mat, key "mask", stored as (ky, kx)
    and transposed to (kx, ky). The lines that are sampled run along kx (the 332-long axis in
    Config.yaml), i.e. mask[0, :, 0] in noise_jiterring is the line pattern.
  * Paper Sec. 4.1: retrospective equispaced undersampling, R = 4, ACS lines kept; Delta_k masks are the
    equispaced pattern shifted by one line while preserving the ACS lines, giving R-1 = 3 masks.
  * nc = 15 coils (Cor-PD knee). Coil maps are smooth and satisfy sum_k |s_k|^2 = 1 (Appendix D).

Deviation (documented): the matrix size is reduced from 332 x 320 (Config.yaml) to a smaller size
(arguments --kx/--ky) so that the 10-unroll ResNet-MoDL finishes on 4 CPU cores within the 2 h budget;
the ACS width is scaled by the same ratio as the matrix (paper: 24 of 332 lines).

Image content is a random sum of ellipses with smooth random phase; all values are random.
"""
import argparse
import os

import numpy as np
import scipy.io as sio


def fft2c(x):
    # same convention as src.Utils.FFT for even sizes: fftshift(fftn(fftshift(x))), ortho
    ax = (-2, -1)
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)


def smooth_noise(rng, shape, cutoff):
    n = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    f = np.fft.fftshift(np.fft.fft2(n))
    yy, xx = np.meshgrid(np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]), indexing="ij")
    f *= np.exp(-(xx**2 + yy**2) / (2 * cutoff**2))
    out = np.fft.ifft2(np.fft.ifftshift(f))
    return out / (np.abs(out).max() + 1e-12)


def make_phantom(rng, nx, ny, textured=True):
    """Complex image, shape (nx, ny); axis 0 is the phase-encode axis (kx in repo naming)."""
    yy, xx = np.meshgrid(np.linspace(-1, 1, ny), np.linspace(-1, 1, nx), indexing="xy")  # (nx, ny)
    img = np.zeros((nx, ny))
    # body outline
    ax_, bx_ = rng.uniform(0.70, 0.85), rng.uniform(0.62, 0.80)
    img += 0.55 * (((xx / bx_) ** 2 + (yy / ax_) ** 2) <= 1.0)
    for _ in range(rng.integers(9, 15)):
        cx, cy = rng.uniform(-0.55, 0.55, 2)
        a, b = rng.uniform(0.04, 0.30, 2)
        th = rng.uniform(0, np.pi)
        xr = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
        yr = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
        inside = ((xr / a) ** 2 + (yr / b) ** 2) <= 1.0
        img += rng.uniform(-0.30, 0.45) * inside
    img = np.clip(img, 0, None)
    if textured:
        img = img * (1.0 + 0.08 * np.real(smooth_noise(rng, (nx, ny), 0.25)))
    img = img / img.max()
    phase = 0.6 * np.real(smooth_noise(rng, (nx, ny), 0.12))
    return img * np.exp(1j * phase)


def make_coils(rng, nc, nx, ny):
    yy, xx = np.meshgrid(np.linspace(-1, 1, ny), np.linspace(-1, 1, nx), indexing="xy")
    coils = np.zeros((nc, nx, ny), dtype=np.complex128)
    for c in range(nc):
        th = 2 * np.pi * c / nc + rng.uniform(-0.1, 0.1)
        px, py = 1.15 * np.sin(th), 1.15 * np.cos(th)
        d2 = (xx - px) ** 2 + (yy - py) ** 2
        mag = np.exp(-d2 / (2 * rng.uniform(0.55, 0.8) ** 2))
        ph = np.pi * (rng.uniform(0.2, 0.6) * (xx * np.sin(th) + yy * np.cos(th))) + rng.uniform(-np.pi, np.pi)
        coils[c] = mag * np.exp(1j * ph)
    sos = np.sqrt((np.abs(coils) ** 2).sum(0, keepdims=True)) + 1e-8
    return coils / sos


def make_masks(kx, acs, R=4):
    """Equispaced R pattern along kx incl. centred ACS; Delta_k = pattern shifted by k lines, ACS kept."""
    c0 = kx // 2
    acs_lo, acs_hi = c0 - acs // 2, c0 + acs // 2
    def pattern(shift):
        m = np.zeros(kx, dtype=np.float32)
        m[(np.arange(kx) + shift) % R == 0] = 1.0
        m[acs_lo:acs_hi] = 1.0
        return m
    omega = pattern(0)
    deltas = [pattern(s) for s in range(1, R)]
    return omega, deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/work/data")
    ap.add_argument("--n-cases", type=int, default=12)
    ap.add_argument("--kx", type=int, default=128)   # phase-encode lines (sampled axis)
    ap.add_argument("--ky", type=int, default=96)
    ap.add_argument("--nc", type=int, default=15)
    ap.add_argument("--noise", type=float, default=0.002, help="k-space noise std relative to peak |k|")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="case")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    acs = int(round(24 * a.kx / 332 / 2) * 2)
    omega, deltas = make_masks(a.kx, acs)
    # masks are stored as (ky, kx) like the repo files (loader transposes to (kx, ky))
    def tile(m):
        return np.tile(m[None, :], (a.ky, 1)).astype(np.float32)
    sio.savemat(os.path.join(a.out, "omega_mask.mat"), {"mask": tile(omega)})
    for i, d in enumerate(deltas, 1):
        sio.savemat(os.path.join(a.out, f"delta{i}.mat"), {"mask": tile(d)})
    for n in range(a.n_cases):
        img = make_phantom(rng, a.kx, a.ky)
        coils = make_coils(rng, a.nc, a.kx, a.ky)
        k = fft2c(img[None] * coils)
        k = k + a.noise * np.abs(k).max() * (rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape)) / np.sqrt(2)
        # stored (nc, ky, kx) so that DataLoaderSL's transpose([0,2,1]) gives (nc, kx, ky)
        ksp = np.ascontiguousarray(k.transpose(0, 2, 1)).astype(np.complex64)
        cs = np.ascontiguousarray(coils.transpose(0, 2, 1)).astype(np.complex64)
        sio.savemat(os.path.join(a.out, f"{a.prefix}_{n:03d}.mat"), {"kspace": ksp, "coils": cs})
    print(f"wrote {a.n_cases} cases: nc={a.nc} kx={a.kx} ky={a.ky} acs={acs} "
          f"omega_lines={int(omega.sum())} delta_lines={[int(d.sum()) for d in deltas]} -> {a.out}")


if __name__ == "__main__":
    main()
