"""Independent implementation of CAME-Grad equations 8-14.

This is not copied from, and cannot substitute for, the withheld official
optimizer. It exists solely to test consequences of the printed equations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class CameGradResult:
    alpha: np.ndarray
    joint: np.ndarray
    mean: np.ndarray
    dual_gradient: np.ndarray
    rectified: np.ndarray
    enhanced: np.ndarray
    final: np.ndarray
    dual_objective: float


def _validate(
    task_gradients: np.ndarray,
    task_weights: np.ndarray,
    rho: float,
    kappa: float,
    nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    gradients = np.asarray(task_gradients, dtype=np.float64)
    weights = np.asarray(task_weights, dtype=np.float64)
    if gradients.ndim != 2 or gradients.shape[0] < 1:
        raise ValueError("task_gradients must have shape (tasks, parameters)")
    if weights.shape != (gradients.shape[0],):
        raise ValueError("task_weights must have one value per task")
    if not np.isfinite(gradients).all() or not np.isfinite(weights).all():
        raise ValueError("inputs must be finite")
    if not 0 <= rho < 1:
        raise ValueError("rho must be in [0, 1)")
    if kappa < 1:
        raise ValueError("kappa must be at least 1")
    if not 0 <= nu <= 1:
        raise ValueError("nu must be in [0, 1]")
    return gradients, weights


def combine_gradients(
    task_gradients: np.ndarray,
    task_weights: np.ndarray,
    *,
    rho: float = 0.5,
    kappa: float = 1.5,
    nu: float = 0.2,
    epsilon: float = 1e-12,
) -> CameGradResult:
    """Apply the paper's three printed stages to a matrix of task gradients.

    A zero dual-gradient makes equation 10 undefined. This implementation
    raises instead of silently inventing an author behavior.
    """

    gradients, weights = _validate(
        task_gradients, task_weights, rho, kappa, nu
    )
    joint = weights @ gradients
    mean = gradients.mean(axis=0)
    radius = rho * np.linalg.norm(mean)

    def objective(alpha: np.ndarray) -> float:
        dual_gradient = alpha @ gradients
        return float(
            dual_gradient @ mean + radius * np.linalg.norm(dual_gradient)
        )

    tasks = gradients.shape[0]
    optimization = minimize(
        objective,
        np.full(tasks, 1.0 / tasks),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * tasks,
        constraints=[
            {
                "type": "eq",
                "fun": lambda alpha: float(np.sum(alpha) - 1.0),
            }
        ],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not optimization.success:
        raise RuntimeError(f"dual optimization failed: {optimization.message}")

    alpha = np.asarray(optimization.x, dtype=np.float64)
    dual_gradient = alpha @ gradients
    dual_norm = np.linalg.norm(dual_gradient)
    if radius > 0 and dual_norm <= epsilon:
        raise ZeroDivisionError(
            "paper equation 10 is undefined when g_alpha has zero norm"
        )
    if radius == 0:
        rectified = mean.copy()
    else:
        rectified = mean + radius * dual_gradient / dual_norm

    target_magnitude = kappa * np.linalg.norm(joint)
    rectified_norm = np.linalg.norm(rectified)
    enhanced = rectified * target_magnitude / (rectified_norm + epsilon)
    final = (1.0 - nu) * enhanced + nu * (kappa * joint)

    return CameGradResult(
        alpha=alpha,
        joint=joint,
        mean=mean,
        dual_gradient=dual_gradient,
        rectified=rectified,
        enhanced=enhanced,
        final=final,
        dual_objective=objective(alpha),
    )
