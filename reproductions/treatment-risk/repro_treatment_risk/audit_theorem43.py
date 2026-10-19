"""Audit the proof step and finite-grid core of Theorem 4.3.

Part 1 gives an exact counterexample to Equation (18): a policy learned on D_m
can attain its empirical constraint with equality while its fresh-sample
expectation is larger. Thus the expectation exchange in the published proof is
not valid for a data-dependent policy.

Part 2 exhaustively checks a repair for an ordered finite tolerance grid. If an
unsafe policy is selected, every preceding upper confidence bound must pass the
prefix constraint. In particular, the first unsafe policy must have an
undercovering UCB. Therefore unsafe selection is contained in one pointwise
UCB-failure event, which restores the alpha guarantee conditional on D_m.

This does not verify the theorem exactly as written over t in (0, 1); it records
the additional finite-grid and conditional-coverage assumptions needed by an
implementable version.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from scipy.stats import binom


F = Fraction


@dataclass
class Equation18Counterexample:
    sample_space: list[int]
    sample_probabilities: list[float]
    tolerance: float
    high_loss: float
    learned_policy_by_sample: list[int]
    empirical_constraint_values: list[float]
    equality_holds_for_every_training_sample: bool
    expected_training_value: float
    fresh_value_by_learned_policy: list[float]
    expected_fresh_value: float
    excess_over_tolerance: float
    equation18_fails: bool


@dataclass
class FiniteGridAudit:
    maximum_grid_size: int
    safety_patterns_checked: int
    ucb_pass_patterns_checked: int
    feasible_candidates_checked: int
    counterexamples_when_first_unsafe_ucb_covers: int
    implication_holds: bool
    implication: str


@dataclass
class BentkusAudit:
    instances: int
    valid_instances: int
    minimum_coverage_margin: float
    tightest_instance: dict[str, float]
    shrink_factor: float
    shrunken_bound_violations: int
    raw_mean_violations: int
    passed: bool


def equation18_counterexample() -> Equation18Counterexample:
    """Two policies interpolate one training atom but fail on a fresh atom."""
    tolerance = F(1, 4)
    high_loss = F(3, 4)
    sample_space = (0, 1)

    # V_j(z) equals tau on the atom j and 3/4 on the other atom.
    value = {
        (policy, z): tolerance if policy == z else high_loss
        for policy in sample_space
        for z in sample_space
    }
    learned = list(sample_space)
    empirical = [value[(z, z)] for z in sample_space]
    fresh_by_policy = [
        sum(value[(policy, z)] for z in sample_space) / 2
        for policy in sample_space
    ]
    expected_training = sum(empirical) / 2
    expected_fresh = sum(fresh_by_policy) / 2
    equality_holds = all(item == tolerance for item in empirical)

    return Equation18Counterexample(
        sample_space=list(sample_space),
        sample_probabilities=[0.5, 0.5],
        tolerance=float(tolerance),
        high_loss=float(high_loss),
        learned_policy_by_sample=learned,
        empirical_constraint_values=[float(item) for item in empirical],
        equality_holds_for_every_training_sample=equality_holds,
        expected_training_value=float(expected_training),
        fresh_value_by_learned_policy=[float(item) for item in fresh_by_policy],
        expected_fresh_value=float(expected_fresh),
        excess_over_tolerance=float(expected_fresh - tolerance),
        equation18_fails=equality_holds and expected_fresh > tolerance,
    )


def audit_finite_grid(maximum_grid_size: int) -> FiniteGridAudit:
    safety_patterns_checked = 0
    ucb_pass_patterns_checked = 0
    feasible_candidates_checked = 0
    counterexamples = 0

    for size in range(1, maximum_grid_size + 1):
        for safe_pattern in itertools.product((False, True), repeat=size):
            safety_patterns_checked += 1
            unsafe_indices = [
                index for index, is_safe in enumerate(safe_pattern) if not is_safe
            ]
            if not unsafe_indices:
                continue
            first_unsafe = unsafe_indices[0]

            # pass_pattern[j] means UCB_j < tau. Coverage at the first unsafe
            # candidate forces pass_pattern[first_unsafe] to be False.
            for pass_pattern in itertools.product((False, True), repeat=size):
                ucb_pass_patterns_checked += 1
                if pass_pattern[first_unsafe]:
                    continue
                prefix_passes = True
                for index in range(size):
                    prefix_passes = prefix_passes and pass_pattern[index]
                    if not prefix_passes:
                        continue
                    feasible_candidates_checked += 1
                    if not safe_pattern[index]:
                        counterexamples += 1

    return FiniteGridAudit(
        maximum_grid_size=maximum_grid_size,
        safety_patterns_checked=safety_patterns_checked,
        ucb_pass_patterns_checked=ucb_pass_patterns_checked,
        feasible_candidates_checked=feasible_candidates_checked,
        counterexamples_when_first_unsafe_ucb_covers=counterexamples,
        implication_holds=counterexamples == 0,
        implication=(
            "unsafe selection implies UCB undercoverage at the first unsafe "
            "candidate on an ordered finite grid"
        ),
    )


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
    return min(
        math.exp(-size * divergence),
        math.e * float(binom.cdf(math.ceil(size * empirical), size, candidate)),
    )


def bentkus_upper_bound(
    empirical: float, size: int, alpha: float, tolerance: float = 1e-12
) -> float:
    low, high = empirical, 1.0
    while high - low > tolerance:
        middle = (low + high) / 2.0
        if bentkus_g(empirical, min(middle, 1.0 - 1e-14), size) >= alpha:
            low = middle
        else:
            high = middle
    return low


def audit_bentkus() -> BentkusAudit:
    probabilities = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    shrink_factor = 0.98
    rows = []
    shrink_violations = 0
    raw_violations = 0
    for size in (5, 8, 10, 20, 50):
        counts = list(range(size + 1))
        for alpha in (0.05, 0.10, 0.20):
            bounds = [
                bentkus_upper_bound(count / size, size, alpha) for count in counts
            ]
            for probability in probabilities:
                masses = binom.pmf(counts, size, probability)
                coverage = float(
                    sum(
                        mass
                        for mass, bound in zip(masses, bounds)
                        if bound >= probability
                    )
                )
                shrunken_coverage = float(
                    sum(
                        mass
                        for mass, bound in zip(masses, bounds)
                        if shrink_factor * bound >= probability
                    )
                )
                raw_coverage = float(
                    sum(
                        mass
                        for mass, count in zip(masses, counts)
                        if count / size >= probability
                    )
                )
                target = 1.0 - alpha
                rows.append(
                    {
                        "n": float(size),
                        "alpha": alpha,
                        "p": probability,
                        "coverage": coverage,
                        "margin": coverage - target,
                    }
                )
                shrink_violations += shrunken_coverage < target
                raw_violations += raw_coverage < target

    tightest = min(rows, key=lambda row: row["margin"])
    valid = sum(row["margin"] >= -1e-12 for row in rows)
    return BentkusAudit(
        instances=len(rows),
        valid_instances=valid,
        minimum_coverage_margin=float(tightest["margin"]),
        tightest_instance=tightest,
        shrink_factor=shrink_factor,
        shrunken_bound_violations=shrink_violations,
        raw_mean_violations=raw_violations,
        passed=valid == len(rows) and shrink_violations > 0 and raw_violations > 0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maximum-grid-size", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    protocol = {
        "equation18": {
            "training_size": 1,
            "sample_space": 2,
            "policy_class": 2,
            "tolerance": "1/4",
            "off_atom_value": "3/4",
            "arithmetic": "fractions.Fraction",
        },
        "finite_grid": {
            "maximum_size": args.maximum_grid_size,
            "enumerates": "all safe/unsafe and UCB pass/fail patterns",
            "prefix_rule": "candidate j feasible only if all UCB[k] < tau for k <= j",
        },
        "bentkus": {
            "sample_sizes": [5, 8, 10, 20, 50],
            "alphas": [0.05, 0.10, 0.20],
            "probabilities": [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
            "enumeration": "all n+1 binomial counts",
            "destructive_controls": ["0.98 * UCB", "raw empirical mean"],
        },
    }
    protocol_json = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    protocol_sha256 = hashlib.sha256(protocol_json.encode("ascii")).hexdigest()

    equation18 = equation18_counterexample()
    finite_grid = audit_finite_grid(args.maximum_grid_size)
    bentkus = audit_bentkus()
    passed = (
        equation18.equation18_fails
        and finite_grid.implication_holds
        and bentkus.passed
    )
    verdict = (
        "PUBLISHED_PROOF_STEP_INVALID; "
        "FINITE_ORDERED_GRID_CERTIFICATE_REPAIRED_CONDITIONALLY"
    )
    result = {
        "claim": "Theorem 4.3 / Equations (13)-(20)",
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "equation18_counterexample": asdict(equation18),
        "finite_grid_audit": asdict(finite_grid),
        "bentkus_audit": asdict(bentkus),
        "verdict": verdict,
        "limitations": [
            "The paper states t in the continuum (0, 1), while the repair assumes an ordered finite grid.",
            "Equation (13) must hold conditional on the independent policy-learning split D_m.",
        ],
        "passed": passed,
    }

    print("Treatment-risk Theorem 4.3 audit")
    print(f"protocol_sha256={protocol_sha256}")
    print(
        "Eq. (18): empirical equality for every training sample="
        f"{equation18.equality_holds_for_every_training_sample}, "
        f"E[training]={equation18.expected_training_value:.3f}, "
        f"E[fresh]={equation18.expected_fresh_value:.3f}, "
        f"fails={equation18.equation18_fails}"
    )
    print(
        f"Finite grid through K={finite_grid.maximum_grid_size}: "
        f"safety_patterns={finite_grid.safety_patterns_checked:,}, "
        f"UCB_patterns={finite_grid.ucb_pass_patterns_checked:,}, "
        "counterexamples under first-unsafe coverage="
        f"{finite_grid.counterexamples_when_first_unsafe_ucb_covers}, "
        f"repair_holds={finite_grid.implication_holds}"
    )
    print(
        f"Bentkus Eq. (15): valid={bentkus.valid_instances}/{bentkus.instances}, "
        f"minimum_margin={bentkus.minimum_coverage_margin:+.6f}, "
        f"0.98x_violations={bentkus.shrunken_bound_violations}, "
        f"raw_mean_violations={bentkus.raw_mean_violations}, "
        f"pass={bentkus.passed}"
    )
    print(f"VERDICT={verdict}")
    print(f"OVERALL_PASS={passed}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
