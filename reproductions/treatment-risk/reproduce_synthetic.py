"""Reproduce the paper's complete observational synthetic experiment.

Implements Eqs. (10), (12), (14), (15), and (21)-(24) for depth-one
fast-and-frugal policies. The run covers four arms:

* no confounding, Gamma=1 (valid);
* no confounding, Gamma=2 (valid and conservative);
* confounding with true odds distortion 2, assumed Gamma=1 (invalid control);
* confounding with true odds distortion 2, assumed Gamma=2 (valid).

The learned policies are evaluated in closed form under the interventional
data-generating process, so reported coverage has no test-set Monte Carlo noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import binom


X_LOW = 30.0
X_HIGH = 80.0
UNTREATED_LOSS = 0.8
UNCONFOUNDED_COEF = 0.01
CONFOUNDED_LOW_U_COEF = 0.02
CONFOUNDED_HIGH_U_COEF = 0.002
U_SPLIT = 0.5
TRAIN_SIZE = 1_000
VALIDATION_SIZE = 1_000
N_BINS = 200
N_T_VALUES = 200
ALPHA = 0.10
T_GRID = np.linspace(0.5 / N_T_VALUES, 0.5, N_T_VALUES)
TAU_GRID = np.arange(0.05, 0.401, 0.05)


@dataclass
class CandidateSet:
    objective: np.ndarray
    constraint: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    treat_below: np.ndarray


@dataclass
class Arm:
    name: str
    gamma: float
    confounded: bool
    gamma_valid: bool
    description: str


def nominal_propensity(x1: np.ndarray | float) -> np.ndarray | float:
    z = 0.5 - (np.asarray(x1) - 30.0) / 50.0
    result = 1.0 / (1.0 + np.exp(-z))
    return float(result) if np.ndim(result) == 0 else result


def draw_data(
    rng: np.random.Generator, size: int, confounded: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x1 = rng.uniform(X_LOW, X_HIGH, size)
    x2 = rng.uniform(X_LOW, X_HIGH, size)
    nominal = nominal_propensity(x1)

    if confounded:
        u = rng.uniform(0.0, 1.0, size)
        assignment_probability = np.where(
            u < U_SPLIT,
            nominal / (2.0 - nominal),
            nominal / (0.5 + 0.5 * nominal),
        )
        treated_loss_probability = np.where(
            u < U_SPLIT,
            CONFOUNDED_LOW_U_COEF,
            CONFOUNDED_HIGH_U_COEF,
        ) * (x1 - 30.0)
    else:
        assignment_probability = nominal
        treated_loss_probability = UNCONFOUNDED_COEF * (x1 - 30.0)

    treatment = rng.random(size) < assignment_probability
    loss_probability = np.where(
        treatment, treated_loss_probability, UNTREATED_LOSS
    )
    loss = rng.random(size) < loss_probability
    return x1, x2, treatment.astype(np.int8), loss.astype(np.int8), nominal


def bin_indices(values: np.ndarray) -> np.ndarray:
    result = np.floor(
        (values - X_LOW) / (X_HIGH - X_LOW) * N_BINS
    ).astype(np.int64)
    return np.clip(result, 0, N_BINS - 1)


def enumerate_candidates(
    x1: np.ndarray,
    x2: np.ndarray,
    treatment: np.ndarray,
    loss: np.ndarray,
    nominal: np.ndarray,
    gamma: float,
) -> CandidateSet:
    observed_probability = np.where(treatment == 1, nominal, 1.0 - nominal)
    weighted_loss = loss * (1.0 + gamma * (1.0 / observed_probability - 1.0))
    edges = np.linspace(X_LOW, X_HIGH, N_BINS + 1)
    parts: dict[str, list[np.ndarray]] = {
        "objective": [],
        "constraint": [],
        "feature": [],
        "threshold": [],
        "treat_below": [],
    }

    for feature, values in enumerate((x1, x2)):
        bins = bin_indices(values)
        action1 = np.bincount(
            bins, weights=weighted_loss * (treatment == 1), minlength=N_BINS
        )
        action0 = np.bincount(
            bins, weights=weighted_loss * (treatment == 0), minlength=N_BINS
        )
        counts = np.bincount(bins, minlength=N_BINS).astype(float)
        prefix1 = np.r_[0.0, np.cumsum(action1)]
        prefix0 = np.r_[0.0, np.cumsum(action0)]
        prefix_count = np.r_[0.0, np.cumsum(counts)]
        total1 = prefix1[-1]
        total0 = prefix0[-1]

        treated_count = prefix_count
        parts["objective"].append(
            (prefix1 + total0 - prefix0) / treatment.size
        )
        parts["constraint"].append(
            np.divide(
                prefix1,
                treated_count,
                out=np.zeros_like(prefix1),
                where=treated_count > 0,
            )
        )
        parts["feature"].append(np.full(edges.size, feature, dtype=np.int8))
        parts["threshold"].append(edges)
        parts["treat_below"].append(np.ones(edges.size, dtype=bool))

        treated_count = treatment.size - prefix_count
        parts["objective"].append(
            (total1 - prefix1 + prefix0) / treatment.size
        )
        parts["constraint"].append(
            np.divide(
                total1 - prefix1,
                treated_count,
                out=np.zeros_like(prefix1),
                where=treated_count > 0,
            )
        )
        parts["feature"].append(np.full(edges.size, feature, dtype=np.int8))
        parts["threshold"].append(edges)
        parts["treat_below"].append(np.zeros(edges.size, dtype=bool))

    return CandidateSet(
        **{name: np.concatenate(values) for name, values in parts.items()}
    )


def policy_mask(
    x1: np.ndarray,
    x2: np.ndarray,
    feature: int,
    threshold: float,
    treat_below: bool,
) -> np.ndarray:
    values = x1 if feature == 0 else x2
    return values < threshold if treat_below else values >= threshold


def bentkus_g(empirical: float, candidate: float, size: int) -> float:
    if candidate <= 0.0 or candidate >= 1.0:
        return 0.0
    divergence = 0.0
    if empirical > 0.0:
        divergence += empirical * math.log(empirical / candidate)
    if empirical < 1.0:
        divergence += (1.0 - empirical) * math.log(
            (1.0 - empirical) / (1.0 - candidate)
        )
    chernoff = math.exp(-size * divergence)
    bentkus = math.e * float(
        binom.cdf(math.ceil(size * empirical), size, candidate)
    )
    return min(chernoff, bentkus)


def bentkus_upper_bound(
    empirical: float, size: int, alpha: float, tolerance: float = 1e-10
) -> float:
    empirical = float(np.clip(empirical, 0.0, 1.0))
    low, high = empirical, 1.0
    while high - low > tolerance:
        middle = (low + high) / 2.0
        if bentkus_g(empirical, min(middle, 1.0 - 1e-12), size) >= alpha:
            low = middle
        else:
            high = middle
    return low


def policy_propensity_infimum(
    feature: int, threshold: float, treat_below: bool
) -> float:
    if feature == 0 and treat_below:
        endpoint = float(np.clip(threshold, X_LOW, X_HIGH))
        return float(nominal_propensity(endpoint))
    return float(nominal_propensity(X_HIGH))


def analytic_treated_fraction(threshold: float, treat_below: bool) -> float:
    threshold = float(np.clip(threshold, X_LOW, X_HIGH))
    below_fraction = (threshold - X_LOW) / (X_HIGH - X_LOW)
    return below_fraction if treat_below else 1.0 - below_fraction


def evaluate_candidate(
    x1: np.ndarray,
    x2: np.ndarray,
    treatment: np.ndarray,
    loss: np.ndarray,
    nominal: np.ndarray,
    gamma: float,
    feature: int,
    threshold: float,
    treat_below: bool,
) -> tuple[float, float]:
    policy = policy_mask(x1, x2, feature, threshold, treat_below)
    observed_probability = np.where(treatment == 1, nominal, 1.0 - nominal)
    weighted_loss = loss * (1.0 + gamma * (1.0 / observed_probability - 1.0))
    objective = float(np.mean(weighted_loss * (treatment == policy)))
    # The paper defines p_pi(A=1) as an identifiable population quantity.
    # It is available exactly for this uniform synthetic DGP and must remain
    # independent of D_n for the conditional UCB guarantee.
    treated_fraction = analytic_treated_fraction(threshold, treat_below)
    if treated_fraction == 0.0:
        return objective, 0.0

    treatment_values = weighted_loss * (treatment == 1) * policy / treated_fraction
    empirical = float(np.mean(treatment_values))
    propensity_infimum = policy_propensity_infimum(
        feature, threshold, treat_below
    )
    maximum = (
        1.0 + gamma * (1.0 / propensity_infimum - 1.0)
    ) / treated_fraction
    upper = maximum * bentkus_upper_bound(
        empirical / maximum, treatment.size, ALPHA
    )
    return objective, min(upper, 1.0)


def true_risks(
    feature: int, threshold: float, treat_below: bool, confounded: bool
) -> tuple[float, float, float]:
    coefficient = (
        U_SPLIT * CONFOUNDED_LOW_U_COEF
        + (1.0 - U_SPLIT) * CONFOUNDED_HIGH_U_COEF
        if confounded
        else UNCONFOUNDED_COEF
    )
    threshold = float(np.clip(threshold, X_LOW, X_HIGH))
    if feature == 0:
        low, high = (
            (X_LOW, threshold) if treat_below else (threshold, X_HIGH)
        )
        treated_fraction = (high - low) / (X_HIGH - X_LOW)
        treated_loss_mass = coefficient * (
            (high - 30.0) ** 2 - (low - 30.0) ** 2
        ) / (2.0 * (X_HIGH - X_LOW))
    else:
        low, high = (
            (X_LOW, threshold) if treat_below else (threshold, X_HIGH)
        )
        treated_fraction = (high - low) / (X_HIGH - X_LOW)
        treated_loss_mass = treated_fraction * coefficient * 25.0

    population_risk = (
        treated_loss_mass + UNTREATED_LOSS * (1.0 - treated_fraction)
    )
    treatment_risk = (
        treated_loss_mass / treated_fraction if treated_fraction > 0.0 else 0.0
    )
    return population_risk, treatment_risk, treated_fraction


def one_run(
    rng: np.random.Generator, gamma: float, confounded: bool
) -> np.ndarray:
    data = draw_data(rng, TRAIN_SIZE + VALIDATION_SIZE, confounded)
    permutation = rng.permutation(TRAIN_SIZE + VALIDATION_SIZE)
    train_index = permutation[:TRAIN_SIZE]
    validation_index = permutation[TRAIN_SIZE:]
    train = tuple(item[train_index] for item in data)
    validation = tuple(item[validation_index] for item in data)

    candidates = enumerate_candidates(*train, gamma)
    objective_order = np.argsort(candidates.objective, kind="stable")
    chosen = np.empty(T_GRID.size, dtype=np.int64)
    for index, tolerance in enumerate(T_GRID):
        feasible = objective_order[
            candidates.constraint[objective_order] <= tolerance
        ]
        chosen[index] = feasible[0]

    unique_candidates = np.unique(chosen)
    validation_result = {
        int(candidate): evaluate_candidate(
            *validation,
            gamma,
            int(candidates.feature[candidate]),
            float(candidates.threshold[candidate]),
            bool(candidates.treat_below[candidate]),
        )
        for candidate in unique_candidates
    }
    validation_objective = np.array(
        [validation_result[int(candidate)][0] for candidate in chosen]
    )
    confidence_bound = np.array(
        [validation_result[int(candidate)][1] for candidate in chosen]
    )

    result = np.empty((TAU_GRID.size, 3))
    for index, tolerance in enumerate(TAU_GRID):
        prefix_feasible = np.logical_and.accumulate(confidence_bound < tolerance)
        feasible_indices = np.flatnonzero(prefix_feasible)
        if feasible_indices.size == 0:
            result[index] = (UNTREATED_LOSS, 0.0, 0.0)
            continue
        selected_index = feasible_indices[
            np.argmin(validation_objective[feasible_indices])
        ]
        selected = chosen[selected_index]
        result[index] = true_risks(
            int(candidates.feature[selected]),
            float(candidates.threshold[selected]),
            bool(candidates.treat_below[selected]),
            confounded,
        )
    return result


def summarize(results: np.ndarray) -> list[dict[str, float | bool]]:
    rows = []
    for index, tolerance in enumerate(TAU_GRID):
        population = results[:, index, 0]
        treatment = results[:, index, 1]
        treated_fraction = results[:, index, 2]
        coverage = float(np.mean(treatment <= tolerance + 1e-12))
        rows.append(
            {
                "tau": float(tolerance),
                "coverage": coverage,
                "controlled": coverage >= 1.0 - ALPHA,
                "mean_population_risk": float(population.mean()),
                "mean_treatment_risk": float(treatment.mean()),
                "treatment_risk_q10": float(np.quantile(treatment, 0.10)),
                "treatment_risk_q50": float(np.quantile(treatment, 0.50)),
                "treatment_risk_q90": float(np.quantile(treatment, 0.90)),
                "mean_treated_fraction": float(treated_fraction.mean()),
                "fraction_treat_none": float(np.mean(treated_fraction == 0.0)),
            }
        )
    return rows


def run_arm(arm: Arm, runs: int, seed: int) -> tuple[np.ndarray, list[dict]]:
    seed_sequence = np.random.SeedSequence(
        [seed, int(arm.gamma * 1_000), int(arm.confounded)]
    )
    children = seed_sequence.spawn(runs)
    results = np.empty((runs, TAU_GRID.size, 3))
    for run, child in enumerate(children):
        results[run] = one_run(
            np.random.default_rng(child), arm.gamma, arm.confounded
        )
        if (run + 1) % 100 == 0 or run + 1 == runs:
            print(f"{arm.name}: {run + 1}/{runs}", flush=True)
    return results, summarize(results)


def self_test() -> dict[str, float | bool]:
    grid = np.linspace(X_LOW, X_HIGH, 1_001)
    nominal = nominal_propensity(grid)
    low_assignment = nominal / (2.0 - nominal)
    high_assignment = nominal / (0.5 + 0.5 * nominal)
    odds = lambda probability: (1.0 - probability) / probability
    low_ratio = odds(low_assignment) / odds(nominal)
    high_ratio = odds(high_assignment) / odds(nominal)

    unconfounded_all = true_risks(0, X_LOW, False, False)[1]
    confounded_all = true_risks(0, X_LOW, False, True)[1]

    size = 200
    probability = 0.3
    mass = binom.pmf(np.arange(size + 1), size, probability)
    certificate_coverage = sum(
        mass[count]
        for count in range(size + 1)
        if bentkus_upper_bound(count / size, size, ALPHA) >= probability
    )
    raw_mean_coverage = sum(
        mass[count]
        for count in range(size + 1)
        if count / size >= probability
    )
    result = {
        "gamma2_low_branch_max_error": float(np.max(np.abs(low_ratio - 2.0))),
        "gamma2_high_branch_max_error": float(
            np.max(np.abs(high_ratio - 0.5))
        ),
        "treat_all_unconfounded": unconfounded_all,
        "treat_all_confounded": confounded_all,
        "bentkus_coverage_n200_p03": float(certificate_coverage),
        "raw_mean_coverage_n200_p03": float(raw_mean_coverage),
    }
    result["passed"] = bool(
        result["gamma2_low_branch_max_error"] < 1e-12
        and result["gamma2_high_branch_max_error"] < 1e-12
        and abs(unconfounded_all - 0.25) < 1e-12
        and abs(confounded_all - 0.275) < 1e-12
        and certificate_coverage >= 1.0 - ALPHA
        and raw_mean_coverage < 1.0 - ALPHA
    )
    return result


def save_figure(summaries: dict[str, list[dict]], output: Path) -> None:
    colors = {
        "unconfounded_gamma1": "#216e39",
        "unconfounded_gamma2": "#58a66f",
        "confounded_gamma2": "#3569a8",
        "confounded_gamma1": "#c43d35",
    }
    labels = {
        "unconfounded_gamma1": "No confounding, Gamma=1",
        "unconfounded_gamma2": "No confounding, Gamma=2",
        "confounded_gamma2": "Confounded, Gamma=2 valid",
        "confounded_gamma1": "Confounded, Gamma=1 invalid",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for name in (
        "unconfounded_gamma1",
        "unconfounded_gamma2",
        "confounded_gamma2",
        "confounded_gamma1",
    ):
        rows = summaries[name]
        tau = [row["tau"] for row in rows]
        axes[0].plot(
            tau,
            [row["coverage"] for row in rows],
            marker="o",
            color=colors[name],
            label=labels[name],
        )
        axes[1].plot(
            tau,
            [row["treatment_risk_q90"] for row in rows],
            marker="o",
            color=colors[name],
            label=labels[name],
        )
    axes[0].axhline(0.9, color="#555555", linestyle="--", label="Required 0.90")
    axes[0].set(xlabel="Tolerance tau", ylabel="Coverage", ylim=(0.0, 1.03))
    axes[1].plot(TAU_GRID, TAU_GRID, color="#555555", linestyle="--", label="q90 = tau")
    axes[1].set(
        xlabel="Tolerance tau",
        ylabel="90th percentile true treatment risk",
        ylim=(0.0, 0.42),
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Treatment-risk synthetic reproduction")
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("treatment-risk/evidence/synthetic_reproduction.json"),
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("treatment-risk/evidence/synthetic_raw.npz"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("treatment-risk/evidence/synthetic_coverage.png"),
    )
    args = parser.parse_args()

    checks = self_test()
    if not checks["passed"]:
        print(json.dumps(checks, indent=2))
        print("SELF_TEST_PASS=False")
        return 1

    arms = [
        Arm(
            "unconfounded_gamma1",
            1.0,
            False,
            True,
            "Exact nominal propensity.",
        ),
        Arm(
            "unconfounded_gamma2",
            2.0,
            False,
            True,
            "Valid but deliberately conservative.",
        ),
        Arm(
            "confounded_gamma1",
            1.0,
            True,
            False,
            "Destructive control: true odds distortion is 2.",
        ),
        Arm(
            "confounded_gamma2",
            2.0,
            True,
            True,
            "True odds distortion is exactly bounded by 2.",
        ),
    ]
    protocol = {
        "paper": "arXiv:2505.08378v1",
        "equations": ["10", "12", "14", "15", "21", "22", "23", "24"],
        "seed": args.seed,
        "runs_per_arm": args.runs,
        "train_size": TRAIN_SIZE,
        "validation_size": VALIDATION_SIZE,
        "t_grid": {"points": N_T_VALUES, "range": "(0, 0.5]"},
        "tau_grid": TAU_GRID.tolist(),
        "bins_per_feature": N_BINS,
        "candidate_count": 4 * (N_BINS + 1),
        "alpha": ALPHA,
        "p_pi": {
            "policy_learning": "empirical analogue on D_m, as in Equation (12)",
            "certificate": "exact analytic population treated mass, independent of D_n",
        },
        "v_max": "tight analytic upper bound over each policy's treated region",
        "policy_search": "all depth-one binned thresholds, both features and orientations",
        "true_risk_evaluation": "closed form under the interventional DGP",
    }
    protocol_json = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    protocol_sha256 = hashlib.sha256(protocol_json.encode("ascii")).hexdigest()

    start = time.perf_counter()
    raw: dict[str, np.ndarray] = {}
    summaries: dict[str, list[dict]] = {}
    arm_metadata: dict[str, dict] = {}
    for arm in arms:
        results, rows = run_arm(arm, args.runs, args.seed)
        raw[arm.name] = results
        summaries[arm.name] = rows
        arm_metadata[arm.name] = {
            **asdict(arm),
            "rows": rows,
            "controlled_tolerances": int(sum(row["controlled"] for row in rows)),
            "minimum_coverage": float(min(row["coverage"] for row in rows)),
        }
        print(
            f"{arm.name}: controlled={arm_metadata[arm.name]['controlled_tolerances']}/8, "
            f"minimum_coverage={arm_metadata[arm.name]['minimum_coverage']:.3f}"
        )

    elapsed = time.perf_counter() - start
    valid_pass = all(
        arm_metadata[arm.name]["controlled_tolerances"] == len(TAU_GRID)
        for arm in arms
        if arm.gamma_valid
    )
    destructive_control_pass = (
        arm_metadata["confounded_gamma1"]["controlled_tolerances"] < len(TAU_GRID)
    )
    passed = valid_pass and destructive_control_pass
    result = {
        "claim": "Section 5.1 synthetic treatment-risk control",
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "self_test": checks,
        "arms": arm_metadata,
        "valid_arms_pass": valid_pass,
        "invalid_gamma_destructive_control_pass": destructive_control_pass,
        "elapsed_seconds": elapsed,
        "passed": passed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.raw_output, **raw)
    save_figure(summaries, args.figure)

    print(f"protocol_sha256={protocol_sha256}")
    print(f"elapsed_seconds={elapsed:.1f}")
    print(f"OVERALL_PASS={passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
