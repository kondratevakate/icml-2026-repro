#!/usr/bin/env python3
"""Run and summarize the pinned author LogAbs implementation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CODE_ROOT = SCRIPT_DIR.parent / "official" / "code"
DEFAULT_LAMBDAS = [
    0.0,
    0.0005,
    0.00075,
    0.001,
    0.002,
    0.005,
    0.0075,
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
]


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def git_revision(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def summarize(
    result: dict[str, Any],
    elapsed_seconds: float,
    revision: str | None,
) -> dict[str, Any]:
    meta = result["meta"]
    methods = meta["method_order"]
    lambdas = np.asarray(meta["lbds"], dtype=float)
    lower_coverage = 1.0 - meta["alpha"] - 0.01
    upper_coverage = 1.0 - meta["alpha"] + 1.0 / (meta["n"] + 1)

    base_coverage = np.asarray(result["base"][0])
    base_size = np.asarray(result["base"][1])
    base_std_population = np.asarray(result["base"][2])
    stcp_coverage = np.asarray(result["StCP"][0])
    stcp_size = np.asarray(result["StCP"][1])
    stcp_std_population = np.asarray(result["StCP"][2])

    repeat_correction = (
        np.sqrt(meta["repeats"] / (meta["repeats"] - 1))
        if meta["repeats"] > 1
        else None
    )
    method_summaries: dict[str, Any] = {}
    for method_index, method in enumerate(methods):
        acceptable = (
            (stcp_coverage[method_index] >= lower_coverage)
            & (stcp_coverage[method_index] <= upper_coverage)
        )
        candidate_indices = np.flatnonzero(acceptable)
        best_index = (
            int(
                candidate_indices[
                    np.argmin(stcp_std_population[method_index, acceptable])
                ]
            )
            if len(candidate_indices)
            else None
        )
        method_summary: dict[str, Any] = {
            "base": {
                "coverage": base_coverage[method_index],
                "mean_size": base_size[method_index],
                "std_population": base_std_population[method_index],
                "std_sample_corrected": (
                    base_std_population[method_index] * repeat_correction
                    if repeat_correction is not None
                    else None
                ),
            },
            "lambda_grid": [
                {
                    "lambda": lambdas[index],
                    "coverage": stcp_coverage[method_index, index],
                    "mean_size": stcp_size[method_index, index],
                    "std_population": stcp_std_population[
                        method_index, index
                    ],
                    "std_sample_corrected": (
                        stcp_std_population[method_index, index]
                        * repeat_correction
                        if repeat_correction is not None
                        else None
                    ),
                    "acceptable_coverage": bool(acceptable[index]),
                }
                for index in range(len(lambdas))
            ],
            "best_acceptable_lambda_index": best_index,
        }
        if best_index is not None:
            base_std = base_std_population[method_index]
            best_std = stcp_std_population[method_index, best_index]
            method_summary["best_acceptable"] = {
                "lambda": lambdas[best_index],
                "coverage": stcp_coverage[method_index, best_index],
                "mean_size": stcp_size[method_index, best_index],
                "std_population": best_std,
                "std_sample_corrected": (
                    best_std * repeat_correction
                    if repeat_correction is not None
                    else None
                ),
                "relative_std_reduction": (
                    1.0 - best_std / base_std
                    if base_std > 0.0
                    else None
                ),
            }
        method_summaries[method] = method_summary

    protocol_strength = (
        "canonical"
        if (
            meta["repeats"] == 50
            and meta["testN"] == 500
            and meta["n"] == 30
            and meta["m"] == 500
        )
        else "reduced"
        if meta["repeats"] >= 10
        else "smoke-only"
    )
    return {
        "paper_claim": {
            "setting": "LogAbs",
            "n": 30,
            "m": 500,
            "nominal_coverage": 0.9,
            "reported_std": {
                "GLCP": {"base": 1.12, "StCP": 0.77},
                "SCC": {"base": 0.98, "StCP": 0.82},
            },
        },
        "run": {
            "protocol_strength": protocol_strength,
            "elapsed_seconds": elapsed_seconds,
            "official_code_revision": revision,
            "python": sys.version,
            "platform": sys.platform,
            "meta": meta,
            "acceptable_coverage_interval": [
                lower_coverage,
                upper_coverage,
            ],
            "std_note": (
                "Released code uses population std. std_sample_corrected "
                "multiplies by sqrt(R/(R-1)) to match the paper definition."
            ),
        },
        "methods": method_summaries,
        "selected_lambda": result.get("selected_lambda"),
        "selected_metrics": result.get("StCP-sel"),
        "claim_verdict": (
            "ELIGIBLE FOR FULL VERDICT"
            if protocol_strength == "canonical"
            else "DIRECTIONAL ONLY"
            if protocol_strength == "reduced"
            else "NO CLAIM VERDICT"
        ),
    }


def parse_lambdas(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-code-root",
        type=Path,
        default=DEFAULT_CODE_ROOT,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--test-points", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--source-n", type=int, default=2000)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--m", type=int, default=500)
    parser.add_argument("--n-grid", type=int, default=50)
    parser.add_argument(
        "--lambdas",
        type=parse_lambdas,
        default=DEFAULT_LAMBDAS,
    )
    parser.add_argument(
        "--with-secondary-baselines",
        action="store_true",
        help="Also run SDCP, PPI, and no-alignment baselines.",
    )
    args = parser.parse_args()

    code_root = args.official_code_root.resolve()
    simulation_dir = code_root / "SimuAnalysis"
    if not (simulation_dir / "core.py").exists():
        raise SystemExit(f"Missing official core.py under {simulation_dir}")

    sys.path.insert(0, str(simulation_dir))
    old_cwd = Path.cwd()
    os.chdir(simulation_dir)
    try:
        from core import run_experiment

        started = time.perf_counter()
        result = run_experiment(
            d=5,
            r=0.5,
            n=args.n,
            m=args.m,
            N=args.source_n,
            gamma_t=1.0,
            gamma_s=1.2,
            dtype="logabs",
            alpha=0.1,
            repeats=args.repeats,
            testN=args.test_points,
            hidden_dim=[50, 100, 100, 50],
            epoches=args.epochs,
            n_grid=args.n_grid,
            lbds=args.lambdas,
            temperature=10.0,
            alpha_tol=0.02,
            run_sdcp=args.with_secondary_baselines,
            run_ppi=args.with_secondary_baselines,
            run_noal=args.with_secondary_baselines,
            run_sel=True,
        )
        elapsed = time.perf_counter() - started
    finally:
        os.chdir(old_cwd)

    evidence = summarize(result, elapsed, git_revision(code_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            evidence,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
