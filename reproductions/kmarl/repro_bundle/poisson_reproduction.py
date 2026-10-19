"""Poisson half of anchored claim A6 for KM-ARL/KM-ADD.

Paper specification (Appendix C.4): Poisson means 1 before and 4 after the
changepoint, GSR with ground-truth likelihood ratio, geometric and uniform
changepoints, and limited/irregular sequence lengths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from kmarl_repro import estimators


RATE0 = 1.0
RATE1 = 4.0
LOG_RATE_RATIO = np.log(RATE1 / RATE0)
LOG_LR_OFFSET = RATE1 - RATE0


def run_gsr_poisson(
    rng: np.random.Generator,
    n_seq: int,
    nu: np.ndarray,
    lengths: np.ndarray,
    threshold: float,
    max_len: int,
) -> np.ndarray:
    log_threshold = np.log(threshold)
    log_r = np.full(n_seq, -np.inf)
    tau = np.full(n_seq, np.inf)
    alive = np.ones(n_seq, dtype=bool)
    for t in range(max_len):
        alive &= t < lengths
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        rate = np.where(t >= nu[idx], RATE1, RATE0)
        x = rng.poisson(rate)
        log_lr = x * LOG_RATE_RATIO - LOG_LR_OFFSET
        log_r[idx] = np.logaddexp(log_r[idx], 0.0) + log_lr
        hit = idx[log_r[idx] >= log_threshold]
        if hit.size:
            tau[hit] = t
            alive[hit] = False
    return tau


def true_metrics(
    rng: np.random.Generator,
    threshold: float,
    n: int,
    arl_max_len: int,
    add_max_len: int,
) -> tuple[float, float, float, float, float]:
    arl_lengths = np.full(n, float(arl_max_len))
    tau_arl = run_gsr_poisson(
        rng, n, np.full(n, np.inf), arl_lengths, threshold, arl_max_len
    )
    nu_add = rng.integers(0, add_max_len // 2, size=n).astype(float)
    add_lengths = np.full(n, float(add_max_len))
    tau_add = run_gsr_poisson(
        rng, n, nu_add, add_lengths, threshold, add_max_len
    )
    finite_arl = np.isfinite(tau_arl)
    finite_add = np.isfinite(tau_add)
    valid_add = finite_add & (tau_add >= nu_add)
    return (
        float(tau_arl[finite_arl].mean()),
        float((tau_add[valid_add] - nu_add[valid_add]).mean()),
        float(finite_arl.mean()),
        float(finite_add.mean()),
        float(valid_add.mean()),
    )


def uniform_dataset(
    rng: np.random.Generator,
    n: int,
    threshold: float,
    t_lo: int,
    t_hi: int,
    prevalence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = (
        rng.integers(t_lo, t_hi + 1, n).astype(float)
        if t_lo < t_hi
        else np.full(n, float(t_lo))
    )
    has_change = rng.random(n) < prevalence
    nu = np.full(n, np.inf)
    nu[has_change] = rng.integers(0, lengths[has_change].astype(int))
    tau = run_gsr_poisson(rng, n, nu, lengths, threshold, t_hi)
    return tau, nu, lengths


def geometric_dataset(
    rng: np.random.Generator,
    n: int,
    threshold: float,
    t_lo: int,
    t_hi: int,
    probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = (
        rng.integers(t_lo, t_hi + 1, n).astype(float)
        if t_lo < t_hi
        else np.full(n, float(t_lo))
    )
    nu = (rng.geometric(probability, n) - 1).astype(float)
    nu[nu >= lengths] = np.inf
    tau = run_gsr_poisson(rng, n, nu, lengths, threshold, t_hi)
    return tau, nu, lengths


def summarize(rows: list[list[float]]) -> dict[str, float]:
    values = np.asarray(rows, dtype=float)
    km_distance = np.hypot(values[:, 0], values[:, 2])
    lb_distance = np.hypot(values[:, 1], values[:, 3])
    return {
        "km_arl_relative_bias_mean": float(values[:, 0].mean()),
        "km_arl_relative_bias_sd": float(values[:, 0].std(ddof=1)),
        "lb_arl_relative_bias_mean": float(np.nanmean(values[:, 1])),
        "km_add_relative_bias_mean": float(np.nanmean(values[:, 2])),
        "lb_add_relative_bias_mean": float(np.nanmean(values[:, 3])),
        "km_curve_distance_mean": float(np.nanmean(km_distance)),
        "lb_curve_distance_mean": float(np.nanmean(lb_distance)),
        "km_curve_beats_lb_fraction": float(np.nanmean(km_distance < lb_distance)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ground-truth-n", type=int, default=20_000)
    args = parser.parse_args()

    log_thresholds = [
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.2,
        4.5,
        4.6,
        4.8,
        5.0,
        5.5,
    ]
    thresholds = [float(np.exp(value)) for value in log_thresholds]
    seeds = list(range(5))
    protocol = {
        "seed_offset": 20260728,
        "log_gsr_thresholds": log_thresholds,
        "ground_truth_n": args.ground_truth_n,
        "ground_truth_arl_max_len": 20_000,
        "ground_truth_add_max_len": 2_000,
        "n_sequences": 10_000,
        "seeds": seeds,
        "figure_13_setups": [
            ["fixed-1000", 1000, 1000, 0.5],
            ["irregular-50-500", 50, 500, 0.5],
        ],
        "pre_stated_success_rule": (
            "In the censoring-active, non-extrapolation region "
            "(0.2*Tmax <= true ARL <= Tmax), KM's normalized ARL-ADD distance "
            "beats LB in at least 80% of seed-cells and its median distance is "
            "at most 75% of LB. Low-censoring points are a null region, not wins."
        ),
    }
    protocol_json = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    protocol_sha256 = hashlib.sha256(protocol_json.encode("ascii")).hexdigest()

    truths = {}
    for threshold_index, threshold in enumerate(thresholds):
        truth_rng = np.random.default_rng(protocol["seed_offset"] + threshold_index)
        (
            true_arl,
            true_add,
            arl_alarm_rate,
            add_alarm_rate,
            add_effective_rate,
        ) = true_metrics(
            truth_rng,
            threshold,
            args.ground_truth_n,
            protocol["ground_truth_arl_max_len"],
            protocol["ground_truth_add_max_len"],
        )
        truths[str(log_thresholds[threshold_index])] = {
            "arl": true_arl,
            "add": true_add,
            "arl_alarm_rate": arl_alarm_rate,
            "add_alarm_rate": add_alarm_rate,
            "add_effective_rate": add_effective_rate,
        }

    results: dict[str, dict[str, dict[str, float]]] = {}
    stress_cells: list[tuple[float, float]] = []
    null_cells: list[tuple[float, float]] = []
    for setup_index, (name, t_lo, t_hi, prevalence) in enumerate(
        protocol["figure_13_setups"]
    ):
        setup_results = {}
        for threshold_index, threshold in enumerate(thresholds):
            truth = truths[str(log_thresholds[threshold_index])]
            rows = []
            for seed in seeds:
                rng = np.random.default_rng(
                    protocol["seed_offset"]
                    + 100_000 * (setup_index + 1)
                    + 1000 * threshold_index
                    + seed
                )
                tau, nu, lengths = uniform_dataset(
                    rng,
                    protocol["n_sequences"],
                    threshold,
                    t_lo,
                    t_hi,
                    prevalence,
                )
                estimates = estimators(tau, nu, lengths)
                rows.append(
                    [
                        (estimates["KM_ARL"] - truth["arl"]) / truth["arl"],
                        (estimates["LB_ARL"] - truth["arl"]) / truth["arl"],
                        (estimates["KM_ADD"] - truth["add"]) / truth["add"],
                        (estimates["LB_ADD"] - truth["add"]) / truth["add"],
                    ]
                )
            summary = summarize(rows)
            summary["true_arl"] = truth["arl"]
            summary["true_add"] = truth["add"]
            summary["extrapolation"] = bool(truth["arl"] > t_hi)
            summary["censoring_active"] = bool(0.2 * t_hi <= truth["arl"] <= t_hi)
            setup_results[str(log_thresholds[threshold_index])] = summary
            values = np.asarray(rows, dtype=float)
            for row in values:
                cell = (
                    float(np.hypot(row[0], row[2])),
                    float(np.hypot(row[1], row[3])),
                )
                if summary["censoring_active"]:
                    stress_cells.append(cell)
                elif not summary["extrapolation"]:
                    null_cells.append(cell)
        results[name] = setup_results

    stress = np.asarray(stress_cells)
    stress_finite = np.isfinite(stress[:, 1])
    win_fraction = float(np.mean(stress[stress_finite, 0] < stress[stress_finite, 1]))
    median_ratio = float(np.median(stress[stress_finite, 0] / stress[stress_finite, 1]))
    null = np.asarray(null_cells)
    null_finite = np.isfinite(null[:, 1])
    null_median_ratio = float(np.median(null[null_finite, 0] / null[null_finite, 1]))
    passed = win_fraction >= 0.8 and median_ratio <= 0.75

    print("Poisson GSR reproduction for anchored claim A6")
    print(f"protocol_sha256={protocol_sha256}")
    print()
    for name, setup in results.items():
        print(name)
        for log_threshold, row in setup.items():
            marker = (
                "EXTRAP"
                if row["extrapolation"]
                else "stress"
                if row["censoring_active"]
                else "null"
            )
            print(
                f"  logthr={float(log_threshold):>3.1f} "
                f"truth=({row['true_arl']:>7.2f},{row['true_add']:>5.2f}) "
                f"curve_dist KM/LB={row['km_curve_distance_mean']:.3f}/"
                f"{row['lb_curve_distance_mean']:.3f} "
                f"wins={row['km_curve_beats_lb_fraction']:.0%} {marker}"
            )
    print(
        f"\nstress seed-cells: KM beats LB={win_fraction:.1%}; "
        f"median distance ratio={median_ratio:.3f}\n"
        f"low-censoring null median distance ratio={null_median_ratio:.3f}"
    )
    print(f"\nOVERALL_PASS={passed}")

    payload = {
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "ground_truth": truths,
        "results": results,
        "checks": {
            "stress_seed_cells": int(len(stress_cells)),
            "finite_lb_stress_seed_cells": int(stress_finite.sum()),
            "stress_km_curve_beats_lb_fraction": win_fraction,
            "stress_median_km_to_lb_distance_ratio": median_ratio,
            "null_seed_cells": int(len(null_cells)),
            "null_median_km_to_lb_distance_ratio": null_median_ratio,
        },
        "passed": passed,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
