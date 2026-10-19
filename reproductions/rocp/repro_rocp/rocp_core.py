"""Paper-faithful finite-label core for Risk-Optimal Conformal Prediction.

This module intentionally does not import the authors' implementation.  It
implements Algorithm 1 and the t=0 convention from Remark 3.2 of
arXiv:2602.00989v3 so that the paper and released code can be audited
independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np


_TOL = 1e-11


@dataclass(frozen=True)
class PointwiseChoice:
    """The plug-in action, threshold, set, and value at one coverage level."""

    t: float
    value: float
    action: int
    theta: float
    label_set: frozenset[int]


class DiscreteROCP:
    """Finite-label implementation of Sections 4.1-4.2 and Algorithm 1.

    ``loss_matrix[y, a]`` is ell(a, y).  Action and selector ties are
    deterministic: the smallest action index and then the largest t are used.
    """

    def __init__(self, loss_matrix: Sequence[Sequence[float]]) -> None:
        matrix = np.asarray(loss_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
            raise ValueError("loss_matrix must have shape (labels >= 2, actions >= 1)")
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
            raise ValueError("loss_matrix must contain finite nonnegative losses")
        self.loss_matrix = matrix
        self.num_labels, self.num_actions = matrix.shape

    def _probabilities(self, probabilities: Sequence[float]) -> np.ndarray:
        probs = np.asarray(probabilities, dtype=float)
        if probs.shape != (self.num_labels,):
            raise ValueError(f"probabilities must have shape ({self.num_labels},)")
        if np.any(probs < 0) or not np.isclose(probs.sum(), 1.0, atol=_TOL):
            raise ValueError("probabilities must be nonnegative and sum to one")
        return probs

    def candidate_t_values(self, probabilities: Sequence[float]) -> tuple[float, ...]:
        """Return every t at which a finite-label quantile can attain a minimum."""

        probs = self._probabilities(probabilities)
        candidates = {0.0, 1.0}
        for action in range(self.num_actions):
            losses = self.loss_matrix[:, action]
            order = np.argsort(losses, kind="mergesort")
            sorted_losses = losses[order]
            cumulative = np.cumsum(probs[order])
            for index, cumulative_probability in enumerate(cumulative):
                if index == len(losses) - 1 or (
                    sorted_losses[index] < sorted_losses[index + 1] - _TOL
                ):
                    candidates.add(float(cumulative_probability))
        return tuple(sorted(candidates))

    @staticmethod
    def _quantile(losses: np.ndarray, probs: np.ndarray, t: float) -> float:
        order = np.argsort(losses, kind="mergesort")
        sorted_losses = losses[order]
        cumulative = np.cumsum(probs[order])
        index = int(np.searchsorted(cumulative, t, side="left"))
        return float(sorted_losses[min(index, len(sorted_losses) - 1)])

    def pointwise_choice(
        self, probabilities: Sequence[float], t: float
    ) -> PointwiseChoice:
        """Compute V_hat, a_hat, theta_hat, and C_hat_0 at fixed x and t."""

        probs = self._probabilities(probabilities)
        if t < -_TOL or t > 1.0 + _TOL:
            raise ValueError("t must lie in [0, 1]")
        t = float(np.clip(t, 0.0, 1.0))

        # Remark 3.2 is explicit: t=0 has theta=M(a) and C(x,0)=Y.
        if t <= _TOL:
            maxima = np.max(self.loss_matrix, axis=0)
            action = int(np.argmin(maxima))
            theta = float(maxima[action])
            return PointwiseChoice(
                t=0.0,
                value=theta,
                action=action,
                theta=theta,
                label_set=frozenset(range(self.num_labels)),
            )

        best_value = np.inf
        best_action = -1
        best_theta = np.nan
        for action in range(self.num_actions):
            losses = self.loss_matrix[:, action]
            theta = self._quantile(losses, probs, t)
            value = t * theta + (1.0 - t) * float(np.max(losses))
            if value < best_value - _TOL:
                best_value = value
                best_action = action
                best_theta = theta

        selected_losses = self.loss_matrix[:, best_action]
        label_set = frozenset(
            int(label)
            for label in np.flatnonzero(selected_losses <= best_theta + _TOL)
        )
        return PointwiseChoice(
            t=t,
            value=float(best_value),
            action=best_action,
            theta=float(best_theta),
            label_set=label_set,
        )

    def selector(self, probabilities: Sequence[float], beta: float) -> PointwiseChoice:
        """Compute g_hat(x,beta), breaking selector ties toward the largest t."""

        if beta < 0:
            raise ValueError("beta must be nonnegative")
        choices = [
            self.pointwise_choice(probabilities, t)
            for t in self.candidate_t_values(probabilities)
        ]
        objectives = np.asarray([choice.value - beta * choice.t for choice in choices])
        best = float(np.min(objectives))
        tied = [
            choice
            for choice, objective in zip(choices, objectives)
            if objective <= best + _TOL
        ]
        return max(tied, key=lambda choice: choice.t)

    def predict_set(
        self, probabilities: Sequence[float], beta: float
    ) -> frozenset[int]:
        return self.selector(probabilities, beta).label_set

    def beta_breakpoints(self, probabilities: Sequence[float]) -> tuple[float, ...]:
        """All nonnegative intersections of the selector's affine objectives."""

        choices = [
            self.pointwise_choice(probabilities, t)
            for t in self.candidate_t_values(probabilities)
        ]
        breakpoints = {0.0}
        for left, right in combinations(choices, 2):
            delta_t = left.t - right.t
            if abs(delta_t) <= _TOL:
                continue
            beta = (left.value - right.value) / delta_t
            if beta >= -_TOL and np.isfinite(beta):
                breakpoints.add(max(0.0, float(beta)))
        return tuple(sorted(breakpoints))

    def empirical_coverage(
        self,
        probabilities: Sequence[Sequence[float]],
        labels: Sequence[int],
        beta: float,
    ) -> float:
        if len(probabilities) != len(labels) or not labels:
            raise ValueError("probabilities and labels must have the same nonzero length")
        hits = sum(
            int(int(label) in self.predict_set(probs, beta))
            for probs, label in zip(probabilities, labels)
        )
        return hits / len(labels)

    def minimal_feasible_beta(
        self,
        probabilities: Sequence[Sequence[float]],
        labels: Sequence[int],
        alpha: float,
    ) -> float:
        """Solve Algorithm 1's one-dimensional constrained problem exactly.

        In a finite label space every selector is piecewise constant in beta.
        Evaluating all affine-objective intersections therefore avoids the
        monotonic-coverage assumption made by a bisection implementation.
        """

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if len(probabilities) != len(labels) or not labels:
            raise ValueError("probabilities and labels must have the same nonzero length")

        candidates = {0.0}
        for probs in probabilities:
            candidates.update(self.beta_breakpoints(probs))

        for beta in sorted(candidates):
            if self.empirical_coverage(probabilities, labels, beta) + _TOL >= 1 - alpha:
                return float(beta)

        # At sufficiently large beta, g_hat=1 and every selected set contains
        # every label.  This guard also protects against floating-point ties.
        beta = max(candidates, default=0.0) + 1.0
        while self.empirical_coverage(probabilities, labels, beta) + _TOL < 1 - alpha:
            beta = 2.0 * beta + 1.0
            if beta > 1e12:
                raise RuntimeError("failed to find a feasible beta")
        return float(beta)

    def algorithm1_prediction_set(
        self,
        calibration_probabilities: Sequence[Sequence[float]],
        calibration_labels: Sequence[int],
        test_probabilities: Sequence[float],
        alpha: float,
    ) -> frozenset[int]:
        """Run the candidate-label loop in Algorithm 1 literally."""

        if len(calibration_probabilities) != len(calibration_labels):
            raise ValueError("calibration probabilities and labels must align")
        result: set[int] = set()
        for candidate_label in range(self.num_labels):
            augmented_probabilities = [
                *calibration_probabilities,
                test_probabilities,
            ]
            augmented_labels = [*calibration_labels, candidate_label]
            beta = self.minimal_feasible_beta(
                augmented_probabilities, augmented_labels, alpha
            )
            if candidate_label in self.predict_set(test_probabilities, beta):
                result.add(candidate_label)
        return frozenset(result)


def robust_closed_form(
    loss_matrix: Sequence[Sequence[float]],
    label_set: Iterable[int],
    alpha: float,
) -> np.ndarray:
    """Evaluate Lemma 2.1 for every action in a finite label space."""

    matrix = np.asarray(loss_matrix, dtype=float)
    labels = frozenset(int(label) for label in label_set)
    if matrix.ndim != 2 or not labels:
        raise ValueError("loss_matrix must be 2D and label_set must be nonempty")
    if not labels.issubset(range(matrix.shape[0])):
        raise ValueError("label_set contains an invalid label")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")

    inside = np.max(matrix[list(labels), :], axis=0)
    outside_labels = [label for label in range(matrix.shape[0]) if label not in labels]
    outside = (
        np.max(matrix[outside_labels, :], axis=0)
        if outside_labels
        else inside.copy()
    )
    return inside + alpha * np.maximum(outside - inside, 0.0)
