from __future__ import annotations

import argparse
import importlib.util
import json
import math
import py_compile
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT / "official" / "repo"
SOURCE_ROOT = PACKAGE_ROOT / "official" / "source"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scalar_fixed_point_check() -> dict[str, Any]:
    """Check both fixed-point equations on an independent scalar quadratic."""
    a, y, q, rho, gamma = 2.5, 1.7, 0.8, 1.3, 0.4
    lam = rho * gamma

    x_dual = a * y / (a + lam * q)
    u_dual = gamma * q * x_dual
    z_dual = x_dual
    data_residual = a * (x_dual - y) + rho * u_dual
    prox_residual = z_dual - (x_dual + u_dual) + gamma * q * z_dual
    objective_residual = a * (x_dual - y) + lam * q * x_dual

    prox_scale = 1.0 / (1.0 + gamma * q)
    x_loose = a * y / (a + rho * (1.0 - prox_scale))
    denoised = prox_scale * x_loose
    loose_equation_residual = a * (x_loose - y) + rho * (x_loose - denoised)
    original_objective_residual = a * (x_loose - y) + lam * q * x_loose
    moreau_objective_residual = a * (x_loose - y) + rho * (x_loose - denoised)

    return {
        "parameters": {"a": a, "y": y, "q": q, "rho": rho, "gamma": gamma},
        "dual": {
            "x": x_dual,
            "u": u_dual,
            "consensus_residual": abs(x_dual - z_dual),
            "data_update_residual": abs(data_residual),
            "prox_update_residual": abs(prox_residual),
            "original_objective_residual": abs(objective_residual),
        },
        "loose": {
            "x": x_loose,
            "denoised": denoised,
            "fixed_point_equation_residual": abs(loose_equation_residual),
            "original_objective_residual": abs(original_objective_residual),
            "moreau_envelope_objective_residual": abs(moreau_objective_residual),
            "interpretation": (
                "The released theorem's algebraic loose fixed-point equation holds, "
                "but for an exact proximal denoiser it is the stationarity equation "
                "of a Moreau-envelope objective, not the original regularized objective."
            ),
        },
    }


def cg_penalty_check() -> dict[str, Any]:
    """Show how the released x update loses its consensus term when w_tik=0."""
    utils = _load_module("official_algorithms_utils", REPO_ROOT / "algorithms" / "utils.py")
    y = torch.tensor([[[[1.5, -0.5], [0.25, 2.0]]]], dtype=torch.float64)

    def identity(x: torch.Tensor) -> torch.Tensor:
        return x

    def acg(x: torch.Tensor, penalty: float) -> torch.Tensor:
        return identity(identity(x)) + penalty * x

    def solve(anchor: torch.Tensor, penalty: float) -> torch.Tensor:
        rhs = y + penalty * anchor
        return utils.cg_uni(acg, rhs, anchor.clone(), rho=penalty, maxiter=20, tol=1e-12)

    anchor_a = torch.zeros_like(y)
    anchor_b = torch.full_like(y, 3.0)
    zero_a = solve(anchor_a, 0.0)
    zero_b = solve(anchor_b, 0.0)
    positive_a = solve(anchor_a, 0.2)
    positive_b = solve(anchor_b, 0.2)

    return {
        "zero_penalty_anchor_difference": float((zero_a - zero_b).abs().max()),
        "positive_penalty_anchor_difference": float((positive_a - positive_b).abs().max()),
        "zero_penalty_solution_error_vs_y": float((zero_a - y).abs().max()),
        "paper_requires_positive_rho": True,
        "release_example_w_tik": 0,
        "paper_reported_lact_w_tik": 1e-5,
    }


def sh_check(seed: int = 2026, trials: int = 128) -> dict[str, Any]:
    """Exercise the released SH code and independently inspect its FFT construction."""
    sh = _load_module("official_sh", REPO_ROOT / "algorithms" / "SH.py")
    torch.manual_seed(seed)
    batch, height, width = 8, 32, 32
    sigma = 1.0
    yy = torch.arange(height, dtype=torch.float32).view(1, 1, height, 1)
    residual = 0.01 * torch.sin(2.0 * math.pi * 3.0 * yy / height)
    residual = residual.expand(batch, 1, height, width).contiguous()
    clean = torch.zeros_like(residual)

    out, info = sh.spectral_homogenization_2d(
        residual, clean, sigma=sigma, smooth_ks=7, eps_ratio=1e-6, aggregate="batch_mean"
    )

    # Reconstruct the released target spectrum and measure its loss of Hermitian
    # symmetry after zero-padded smoothing in native FFT order.
    r_fft = torch.fft.fft2(residual, dim=(-2, -1))
    power = (r_fft.real**2 + r_fft.imag**2).mean(dim=(0, 1), keepdim=True)
    smoothed = F.avg_pool2d(power, kernel_size=7, stride=1, padding=3)
    target = sigma**2 * height * width
    complement = (target - smoothed).clamp_min(0.0).clamp_min(1e-6 * target)
    indices_h = torch.remainder(-torch.arange(height), height)
    indices_w = torch.remainder(-torch.arange(width), width)
    reflected = complement.index_select(-2, indices_h).index_select(-1, indices_w)
    hermitian_asymmetry = float((complement - reflected).abs().max() / complement.mean())

    # Estimate the actual expected effective spectrum produced after the code
    # discards the imaginary part of a generally non-Hermitian inverse FFT.
    accumulated = torch.zeros((height, width), dtype=torch.float64)
    for _ in range(trials):
        sample, _ = sh.spectral_homogenization_2d(
            residual[:1], clean[:1], sigma=sigma, smooth_ks=7, eps_ratio=1e-6, aggregate="batch_mean"
        )
        sample_fft = torch.fft.fft2(sample, dim=(-2, -1))
        accumulated += (sample_fft.real**2 + sample_fft.imag**2)[0, 0].double()
    mean_effective_power = accumulated / trials
    relative_rmse_to_white = float(
        torch.sqrt(torch.mean((mean_effective_power - target) ** 2)) / target
    )
    peak_ratio = float(mean_effective_power.max() / target)
    valley_ratio = float(mean_effective_power.min() / target)

    return {
        "shape_preserved": list(out.shape) == list(residual.shape),
        "dtype_preserved": str(out.dtype) == str(residual.dtype),
        "finite": bool(torch.isfinite(out).all()),
        "reported_info": info,
        "complement_hermitian_asymmetry_ratio": hermitian_asymmetry,
        "effective_psd_relative_rmse_to_white": relative_rmse_to_white,
        "effective_psd_peak_ratio": peak_ratio,
        "effective_psd_valley_ratio": valley_ratio,
        "trials": trials,
        "paper_vs_code_sampling": (
            "The paper uses sqrt(Delta S) times random phase; code multiplies "
            "the full random FFT by sqrt(Delta S / HW) and then drops the "
            "imaginary inverse-FFT component."
        ),
    }


def release_inventory() -> dict[str, Any]:
    tracked = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "ls-tree", "-r", "--name-only", "HEAD"],
        text=True,
        encoding="utf-8",
    ).splitlines()
    py_files = sorted(REPO_ROOT.rglob("*.py"))
    compile_errors: dict[str, str] = {}
    for path in py_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            compile_errors[str(path.relative_to(REPO_ROOT))] = str(exc)

    source = (SOURCE_ROOT / "main.tex").read_text(encoding="utf-8")
    shell = (REPO_ROOT / "recon_PBCT.sh").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    return {
        "tracked_file_count": len(tracked),
        "python_file_count": len(py_files),
        "python_compile_errors": compile_errors,
        "has_mri_implementation": any("mri" in path.lower() or "fast" in path.lower() for path in tracked),
        "has_cached_outputs": any(
            (
                any(part.lower() in {"results", "outputs"} for part in Path(path).parts)
                or Path(path).name.lower().startswith("metrics_summary")
                or Path(path).suffix.lower() in {".nii", ".pkl", ".pt", ".pth"}
                or path.lower().endswith(".nii.gz")
            )
            for path in tracked
        ),
        "has_tests": any("test" in Path(path).name.lower() for path in tracked),
        "has_requirements": any(Path(path).name.lower().startswith("requirement") for path in tracked),
        "example_method": "DiffPIR" if "METHOD=DiffPIR" in shell else "unknown",
        "readme_quickstart_w_tik_zero": "--w-tik 0" in readme,
        "paper_lact_rho_1e_minus_5": (
            r"\rho = 1 \times 10^{-5}\times \frac{1}{\sigma_t^2}" in source
        ),
        "theorem_scaling_inconsistency": {
            "main_theorem_omits_effective_lambda": (
                r"\nabla f(\mathbf{x}^*) + \partial \phi(\mathbf{x}^*)" in source
            ),
            "appendix_sets_lambda_rho_gamma": r"\lambda = \rho \gamma" in source,
            "loose_statement_uses_rho_over_sigma2": r"\frac{\rho}{\sigma^2} \mathcal{E}_{bias}" in source,
            "loose_proof_uses_rho_times_residual": (
                r"\rho(\tilde{\mathbf{x}} - \mathcal{D}_\sigma(\tilde{\mathbf{x}}))" in source
            ),
        },
    }


def run_audit(seed: int = 2026, trials: int = 128) -> dict[str, Any]:
    return {
        "paper": {
            "title": (
                "Plug-and-Play Diffusion Meets ADMM: Dual-Variable Coupling "
                "for Robust Medical Image Reconstruction"
            ),
            "openreview_id": "jEBkuuETjr",
            "arxiv": "2602.23214v2",
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "fixed_points": scalar_fixed_point_check(),
        "released_cg_penalty": cg_penalty_check(),
        "spectral_homogenization": sh_check(seed=seed, trials=trials),
        "release_inventory": release_inventory(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "evidence" / "audit.json",
    )
    args = parser.parse_args()
    result = run_audit(seed=args.seed, trials=args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
