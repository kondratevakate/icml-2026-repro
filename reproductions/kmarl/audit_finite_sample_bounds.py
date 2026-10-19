"""Independent numerical audit of Theorems 4.1 and 4.2.

The two theorems have the same mathematical form.  For independent continuous
event and censoring times, they bound the finite-sample bias of the truncated
Kaplan-Meier mean by

    -int_0^a t G(t) H(t)^(N-1) dF(t)
        <= E[int_0^a Shat(t) dt] - int_0^a S(t) dt
        <= int_0^a a G(t) H(t)^(N-1) dF(t).

This audit uses exponential event/censoring laws, evaluates both integrals with
Gauss-Legendre quadrature, and estimates the expectation of the KM mean by
independent Monte Carlo.  A destructive control makes censoring deterministically
depend on the event time; the theorem's marginal-distribution bounds must then
fail because the independent-censoring assumption has been removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class AuditRow:
    label: str
    n: int
    lower: float
    bias: float
    upper: float
    standard_error: float
    inside_5se: bool


def km_areas(event: np.ndarray, censor: np.ndarray, upper: float) -> np.ndarray:
    """Vectorized area under KM survival for one batch of continuous samples."""
    n_trials, n = event.shape
    observed = event < censor
    exit_time = np.minimum(event, censor)
    order = np.argsort(exit_time, axis=1)
    times = np.take_along_axis(exit_time, order, axis=1)
    deaths = np.take_along_axis(observed, order, axis=1)

    survival = np.ones(n_trials)
    previous = np.zeros(n_trials)
    area = np.zeros(n_trials)
    for j in range(n):
        current = times[:, j]
        before_upper = current < upper
        endpoint = np.minimum(current, upper)
        area += survival * np.maximum(endpoint - previous, 0.0)
        risk = n - j
        survival *= np.where(before_upper & deaths[:, j], 1.0 - 1.0 / risk, 1.0)
        previous = endpoint
    area += survival * np.maximum(upper - previous, 0.0)
    return area


def theorem_bounds(
    n: int,
    event_rate: float,
    censor_rate: float,
    upper: float,
    quadrature_order: int = 256,
) -> tuple[float, float]:
    """Evaluate the paper's lower and upper bounds for exponential marginals."""
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    t = 0.5 * upper * (nodes + 1.0)
    w = 0.5 * upper * weights
    g = 1.0 - np.exp(-censor_rate * t)
    h = 1.0 - np.exp(-(event_rate + censor_rate) * t)
    d_f = event_rate * np.exp(-event_rate * t)
    common = g * np.power(h, n - 1) * d_f
    return -float(np.sum(w * t * common)), float(np.sum(w * upper * common))


def dependent_bounds(
    n: int,
    event_rate: float,
    censor_fraction: float,
    upper: float,
    quadrature_order: int = 256,
) -> tuple[float, float]:
    """Plug dependent-control marginals into the theorem as a destructive test."""
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    t = 0.5 * upper * (nodes + 1.0)
    w = 0.5 * upper * weights
    censor_rate = event_rate / censor_fraction
    g = 1.0 - np.exp(-censor_rate * t)
    h = g  # censor_fraction < 1 means min(event, censor) is always censor
    d_f = event_rate * np.exp(-event_rate * t)
    common = g * np.power(h, n - 1) * d_f
    return -float(np.sum(w * t * common)), float(np.sum(w * upper * common))


def monte_carlo_bias(
    rng: np.random.Generator,
    n: int,
    event_rate: float,
    censor_rate: float,
    upper: float,
    trials: int,
    batch_size: int,
) -> tuple[float, float]:
    values: list[np.ndarray] = []
    remaining = trials
    while remaining:
        batch = min(batch_size, remaining)
        event = rng.exponential(1.0 / event_rate, size=(batch, n))
        censor = rng.exponential(1.0 / censor_rate, size=(batch, n))
        values.append(km_areas(event, censor, upper))
        remaining -= batch
    areas = np.concatenate(values)
    target = (1.0 - np.exp(-event_rate * upper)) / event_rate
    return float(areas.mean() - target), float(areas.std(ddof=1) / np.sqrt(trials))


def run_context(
    label: str,
    seed: int,
    event_rate: float,
    censor_rate: float,
    upper: float,
    sample_sizes: list[int],
    trials: int,
    batch_size: int,
) -> list[AuditRow]:
    rng = np.random.default_rng(seed)
    rows = []
    for n in sample_sizes:
        lower, upper_bound = theorem_bounds(n, event_rate, censor_rate, upper)
        bias, standard_error = monte_carlo_bias(
            rng, n, event_rate, censor_rate, upper, trials, batch_size
        )
        rows.append(
            AuditRow(
                label=label,
                n=n,
                lower=lower,
                bias=bias,
                upper=upper_bound,
                standard_error=standard_error,
                inside_5se=(bias - 5 * standard_error >= lower)
                and (bias + 5 * standard_error <= upper_bound),
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sample_sizes = [2, 4, 8, 16, 32, 64]
    protocol = {
        "seed": 20260728,
        "trials_per_cell": args.trials,
        "sample_sizes": sample_sizes,
        "arl": {"event_rate": 0.3, "censor_rate": 0.1, "upper": 10.0},
        "add": {"event_rate": 0.5, "censor_rate": 0.3, "upper": 5.0},
        "destructive_control": {
            "event_rate": 0.3,
            "censor_fraction": 0.5,
            "upper": 10.0,
            "n": 64,
        },
    }
    protocol_json = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    protocol_sha256 = hashlib.sha256(protocol_json.encode("ascii")).hexdigest()

    rows = []
    rows.extend(
        run_context(
            "ARL / Theorem 4.1",
            protocol["seed"],
            **protocol["arl"],
            sample_sizes=sample_sizes,
            trials=args.trials,
            batch_size=args.batch_size,
        )
    )
    rows.extend(
        run_context(
            "ADD / Theorem 4.2",
            protocol["seed"] + 1,
            **protocol["add"],
            sample_sizes=sample_sizes,
            trials=args.trials,
            batch_size=args.batch_size,
        )
    )

    mutation = protocol["destructive_control"]
    n = mutation["n"]
    lower, upper_bound = dependent_bounds(**mutation)
    target = (1.0 - np.exp(-mutation["event_rate"] * mutation["upper"])) / mutation[
        "event_rate"
    ]
    dependent_bias = mutation["upper"] - target
    mutation_fails = not (lower <= dependent_bias <= upper_bound)

    print("Finite-sample KM bias bound audit")
    print(f"protocol_sha256={protocol_sha256}")
    print(f"trials_per_cell={args.trials:,}")
    print()
    print(f"{'context':<23} {'N':>3} {'lower':>11} {'bias':>11} {'upper':>11} {'SE':>10} pass")
    for row in rows:
        print(
            f"{row.label:<23} {row.n:>3} {row.lower:>+11.6f} "
            f"{row.bias:>+11.6f} {row.upper:>+11.6f} "
            f"{row.standard_error:>10.6f} {row.inside_5se}"
        )
    print()
    print("Destructive control: censor = 0.5 * event (independence removed)")
    print(
        f"N={n} marginal-bound=[{lower:+.6f}, {upper_bound:+.6f}] "
        f"observed_bias={dependent_bias:+.6f} bound_violated={mutation_fails}"
    )

    passed = all(row.inside_5se for row in rows) and mutation_fails
    result = {
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "rows": [asdict(row) for row in rows],
        "destructive_control": {
            "lower": lower,
            "bias": dependent_bias,
            "upper": upper_bound,
            "bound_violated": mutation_fails,
        },
        "passed": passed,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nOVERALL_PASS={passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
