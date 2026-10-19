"""Exact finite-state audit of Lemma 4.1.

The paper bounds population and treatment risks by importance-weighted
expectations under either observational treatment assignment or randomized
trial selection. This script enumerates finite binary (X, U, A, L) models,
all four deterministic policies on binary X, and all 256 deterministic
potential-loss tables. All probability calculations use Fraction arithmetic.

The destructive controls generate data with odds distortion Gamma=2 but
evaluate the paper's weights with Gamma=1. They are expected to violate the
claimed upper bound because the stated sensitivity assumption is then false.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable


F = Fraction
State = tuple[int, int]
Policy = tuple[int, int]
LossTable = tuple[int, ...]


@dataclass
class ScenarioResult:
    scenario: str
    models: int
    policies_per_model: int
    loss_tables_per_policy: int
    risk_checks: int
    treatment_checks: int
    maximum_risk_excess: float
    maximum_treatment_excess: float
    maximum_gamma1_equality_error: float
    passed: bool


@dataclass
class MutationResult:
    scenario: str
    true_gamma: float
    assumed_gamma: float
    policy: list[int]
    loss_table: list[int]
    true_risk: float
    claimed_risk_bound: float
    risk_excess: float
    true_treatment_risk: float | None
    claimed_treatment_bound: float | None
    treatment_excess: float | None
    bound_violated: bool


def odds_to_probability(odds: F) -> F:
    return F(1, 1) / (F(1, 1) + odds)


def loss_at(loss: LossTable, x: int, u: int, a: int) -> int:
    return loss[4 * x + 2 * u + a]


def policies() -> Iterable[Policy]:
    return itertools.product((0, 1), repeat=2)


def loss_tables() -> Iterable[LossTable]:
    return itertools.product((0, 1), repeat=8)


def state_distribution(p_x1: F, p_u1_x0: F, p_u1_x1: F) -> dict[State, F]:
    result: dict[State, F] = {}
    for x in (0, 1):
        p_x = p_x1 if x else 1 - p_x1
        p_u1 = p_u1_x1 if x else p_u1_x0
        for u in (0, 1):
            result[(x, u)] = p_x * (p_u1 if u else 1 - p_u1)
    assert sum(result.values()) == 1
    return result


def assignment_probabilities(
    nominal: tuple[F, F],
    gamma: F,
    pattern: tuple[int, int, int, int],
) -> dict[State, F]:
    result = {}
    for x, u in itertools.product((0, 1), repeat=2):
        nominal_odds = (1 - nominal[x]) / nominal[x]
        multiplier = (F(1, 1) / gamma, F(1, 1), gamma)[pattern[2 * x + u]]
        result[(x, u)] = odds_to_probability(multiplier * nominal_odds)
    return result


def observational_metrics(
    distribution: dict[State, F],
    assignment: dict[State, F],
    nominal: tuple[F, F],
    assumed_gamma: F,
    policy: Policy,
    loss: LossTable,
) -> tuple[F, F | None, F, F | None]:
    risk = F(0)
    risk_bound = F(0)
    treated_probability = sum(
        probability for (x, _), probability in distribution.items() if policy[x] == 1
    )
    treatment_numerator = F(0)
    treatment_bound_numerator = F(0)

    for (x, u), state_probability in distribution.items():
        target_a = policy[x]
        target_loss = F(loss_at(loss, x, u, target_a))
        risk += state_probability * target_loss
        if target_a == 1:
            treatment_numerator += state_probability * target_loss

        for a in (0, 1):
            p_a = assignment[(x, u)] if a else 1 - assignment[(x, u)]
            if a != target_a:
                continue
            nominal_a = nominal[x] if a else 1 - nominal[x]
            weight = 1 + assumed_gamma * (1 / nominal_a - 1)
            observed_term = state_probability * p_a * F(loss_at(loss, x, u, a))
            risk_bound += observed_term * weight
            if a == 1:
                treatment_bound_numerator += observed_term * weight

    if treated_probability == 0:
        return risk, None, risk_bound, None
    return (
        risk,
        treatment_numerator / treated_probability,
        risk_bound,
        treatment_bound_numerator / treated_probability,
    )


def trial_conditionals(
    population: dict[State, F],
    selection: dict[State, F],
) -> tuple[dict[State, F], dict[State, F], F, F]:
    p_s1 = sum(population[state] * selection[state] for state in population)
    p_s0 = 1 - p_s1
    study = {
        state: population[state] * selection[state] / p_s1 for state in population
    }
    target = {
        state: population[state] * (1 - selection[state]) / p_s0
        for state in population
    }
    return target, study, p_s0, p_s1


def trial_metrics(
    population: dict[State, F],
    selection: dict[State, F],
    nominal_selection: tuple[F, F],
    treatment_probability: tuple[F, F],
    assumed_gamma: F,
    policy: Policy,
    loss: LossTable,
) -> tuple[F, F | None, F, F | None]:
    target, study, p_s0, p_s1 = trial_conditionals(population, selection)
    risk = F(0)
    risk_bound = F(0)
    treated_probability = sum(
        probability for (x, _), probability in target.items() if policy[x] == 1
    )
    treatment_numerator = F(0)
    treatment_bound_numerator = F(0)

    for (x, u), target_probability in target.items():
        target_a = policy[x]
        target_loss = F(loss_at(loss, x, u, target_a))
        risk += target_probability * target_loss
        if target_a == 1:
            treatment_numerator += target_probability * target_loss

    for (x, u), study_probability in study.items():
        nominal_odds = (1 - nominal_selection[x]) / nominal_selection[x]
        selection_weight = assumed_gamma * nominal_odds * p_s1 / p_s0
        for a in (0, 1):
            p_a = treatment_probability[x] if a else 1 - treatment_probability[x]
            if a != policy[x]:
                continue
            observed_term = study_probability * p_a * F(loss_at(loss, x, u, a))
            policy_weight = selection_weight / p_a
            risk_bound += observed_term * policy_weight
            if a == 1:
                treatment_bound_numerator += observed_term * policy_weight

    if treated_probability == 0:
        return risk, None, risk_bound, None
    return (
        risk,
        treatment_numerator / treated_probability,
        risk_bound,
        treatment_bound_numerator / treated_probability,
    )


def representative_patterns(gamma: F) -> list[tuple[int, int, int, int]]:
    if gamma == 1:
        return [(1, 1, 1, 1)]
    return [
        (0, 0, 0, 0),
        (2, 2, 2, 2),
        (0, 2, 0, 2),
        (2, 0, 2, 0),
        (0, 2, 2, 0),
        (2, 0, 0, 2),
        (0, 1, 2, 1),
        (2, 1, 0, 1),
    ]


def audit_scenario(
    scenario: str,
    metric_factory: Callable[
        [F, tuple[F, F], tuple[int, int, int, int]],
        Callable[[Policy, LossTable], tuple[F, F | None, F, F | None]],
    ],
) -> ScenarioResult:
    model_count = 0
    risk_checks = 0
    treatment_checks = 0
    max_risk_excess = F(0)
    max_treatment_excess = F(0)
    max_equality_error = F(0)

    nominal_pairs = [(F(1, 4), F(3, 4)), (F(1, 3), F(2, 3)), (F(1, 2), F(1, 2))]
    for gamma in (F(1), F(3, 2), F(2), F(3)):
        for nominal in nominal_pairs:
            for pattern in representative_patterns(gamma):
                metrics = metric_factory(gamma, nominal, pattern)
                model_count += 1
                for policy in policies():
                    for loss in loss_tables():
                        risk, treatment, risk_bound, treatment_bound = metrics(policy, loss)
                        max_risk_excess = max(max_risk_excess, risk - risk_bound)
                        risk_checks += 1
                        if gamma == 1:
                            max_equality_error = max(
                                max_equality_error, abs(risk - risk_bound)
                            )
                        if treatment is not None and treatment_bound is not None:
                            max_treatment_excess = max(
                                max_treatment_excess, treatment - treatment_bound
                            )
                            treatment_checks += 1
                            if gamma == 1:
                                max_equality_error = max(
                                    max_equality_error,
                                    abs(treatment - treatment_bound),
                                )

    passed = (
        max_risk_excess <= 0
        and max_treatment_excess <= 0
        and max_equality_error == 0
    )
    return ScenarioResult(
        scenario=scenario,
        models=model_count,
        policies_per_model=4,
        loss_tables_per_policy=256,
        risk_checks=risk_checks,
        treatment_checks=treatment_checks,
        maximum_risk_excess=float(max_risk_excess),
        maximum_treatment_excess=float(max_treatment_excess),
        maximum_gamma1_equality_error=float(max_equality_error),
        passed=passed,
    )


def observational_factory(
    gamma: F,
    nominal: tuple[F, F],
    pattern: tuple[int, int, int, int],
) -> Callable[[Policy, LossTable], tuple[F, F | None, F, F | None]]:
    distribution = state_distribution(F(2, 5), F(1, 4), F(3, 4))
    assignment = assignment_probabilities(nominal, gamma, pattern)
    return lambda policy, loss: observational_metrics(
        distribution, assignment, nominal, gamma, policy, loss
    )


def trial_factory(
    gamma: F,
    nominal: tuple[F, F],
    pattern: tuple[int, int, int, int],
) -> Callable[[Policy, LossTable], tuple[F, F | None, F, F | None]]:
    population = state_distribution(F(3, 5), F(2, 3), F(1, 3))
    selection = assignment_probabilities(nominal, gamma, pattern)
    treatment_probability = (F(1, 3), F(2, 3))
    return lambda policy, loss: trial_metrics(
        population,
        selection,
        nominal,
        treatment_probability,
        gamma,
        policy,
        loss,
    )


def find_mutation(
    scenario: str,
    metrics: Callable[[Policy, LossTable], tuple[F, F | None, F, F | None]],
    require_treatment: bool = True,
) -> MutationResult:
    best: tuple[F, Policy, LossTable, tuple[F, F | None, F, F | None]] | None = None
    for policy in policies():
        for loss in loss_tables():
            values = metrics(policy, loss)
            risk, treatment, risk_bound, treatment_bound = values
            if require_treatment and (treatment is None or treatment_bound is None):
                continue
            if treatment is not None and treatment_bound is not None:
                excess = treatment - treatment_bound
            else:
                excess = risk - risk_bound
            if best is None or excess > best[0]:
                best = (excess, policy, loss, values)
    assert best is not None
    excess, policy, loss, values = best
    risk, treatment, risk_bound, treatment_bound = values
    return MutationResult(
        scenario=scenario,
        true_gamma=2.0,
        assumed_gamma=1.0,
        policy=list(policy),
        loss_table=list(loss),
        true_risk=float(risk),
        claimed_risk_bound=float(risk_bound),
        risk_excess=float(risk - risk_bound),
        true_treatment_risk=None if treatment is None else float(treatment),
        claimed_treatment_bound=(
            None if treatment_bound is None else float(treatment_bound)
        ),
        treatment_excess=(
            None
            if treatment is None or treatment_bound is None
            else float(treatment - treatment_bound)
        ),
        bound_violated=excess > 0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    protocol = {
        "state_space": {"X": 2, "U": 2, "A": 2, "potential_loss_cells": 8},
        "policies": 4,
        "loss_tables": 256,
        "gammas": ["1", "3/2", "2", "3"],
        "nominal_probability_pairs": [["1/4", "3/4"], ["1/3", "2/3"], ["1/2", "1/2"]],
        "odds_patterns": "1 at Gamma=1; 8 representative boundary/interior patterns otherwise",
        "arithmetic": "fractions.Fraction",
    }
    protocol_json = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    protocol_sha256 = hashlib.sha256(protocol_json.encode("ascii")).hexdigest()

    observational = audit_scenario("observational / Eq. (10)", observational_factory)
    trial = audit_scenario("randomized trial / Eq. (11)", trial_factory)

    true_gamma = F(2)
    assumed_gamma = F(1)
    destructive_pattern = (2, 2, 2, 2)

    obs_nominal = (F(1, 3), F(2, 3))
    obs_distribution = state_distribution(F(2, 5), F(1, 4), F(3, 4))
    obs_assignment = assignment_probabilities(
        obs_nominal, true_gamma, destructive_pattern
    )
    obs_mutation = find_mutation(
        "observational / underestimated Gamma",
        lambda policy, loss: observational_metrics(
            obs_distribution,
            obs_assignment,
            obs_nominal,
            assumed_gamma,
            policy,
            loss,
        ),
    )

    trial_nominal = (F(1, 3), F(2, 3))
    trial_population = state_distribution(F(3, 5), F(2, 3), F(1, 3))
    trial_selection = assignment_probabilities(
        trial_nominal, true_gamma, destructive_pattern
    )
    trial_mutation = find_mutation(
        "randomized trial / underestimated Gamma",
        lambda policy, loss: trial_metrics(
            trial_population,
            trial_selection,
            trial_nominal,
            (F(1, 3), F(2, 3)),
            assumed_gamma,
            policy,
            loss,
        ),
    )

    scenarios = [observational, trial]
    mutations = [obs_mutation, trial_mutation]
    passed = all(row.passed for row in scenarios) and all(
        row.bound_violated for row in mutations
    )
    result = {
        "claim": "Lemma 4.1 / Equations (9)-(11)",
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "scenarios": [asdict(row) for row in scenarios],
        "destructive_controls": [asdict(row) for row in mutations],
        "passed": passed,
    }

    print("Treatment-risk Lemma 4.1 finite-state audit")
    print(f"protocol_sha256={protocol_sha256}")
    for row in scenarios:
        print(
            f"{row.scenario}: models={row.models}, "
            f"risk_checks={row.risk_checks:,}, "
            f"treatment_checks={row.treatment_checks:,}, "
            f"max_risk_excess={row.maximum_risk_excess:.3g}, "
            f"max_treatment_excess={row.maximum_treatment_excess:.3g}, "
            f"gamma1_equality_error={row.maximum_gamma1_equality_error:.3g}, "
            f"pass={row.passed}"
        )
    for row in mutations:
        print(
            f"{row.scenario}: risk_excess={row.risk_excess:+.6f}, "
            f"treatment_excess={row.treatment_excess!s}, "
            f"violated={row.bound_violated}"
        )
    print(f"OVERALL_PASS={passed}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
