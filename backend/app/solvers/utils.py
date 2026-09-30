from __future__ import annotations

import numpy as np

from ..problem_definition import ProblemDefinition
from ..schemas import Iterate


def round_vector(vector: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in np.ravel(vector)]


def round_matrix(matrix: np.ndarray) -> list[list[float]]:
    return [[round(float(value), 6) for value in row] for row in matrix]


def iterate_point(
    problem: ProblemDefinition,
    iteration: int,
    x: np.ndarray,
    step_norm: float = 0.0,
) -> Iterate:
    return Iterate(
        iteration=iteration,
        x=round_vector(x),
        objective=round(problem.objective(x), 8),
        violation=round(problem.violation(x), 8),
        step_norm=round(step_norm, 8),
    )


def safe_solve(hessian: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    regularized = hessian + 1e-8 * np.eye(hessian.shape[0])
    try:
        return np.linalg.solve(regularized, gradient)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(regularized) @ gradient


def grid_search_minimize(
    problem: ProblemDefinition,
    resolution: int = 120,
) -> np.ndarray:
    x_values = np.linspace(problem.bounds[0][0], problem.bounds[0][1], resolution)
    y_values = np.linspace(problem.bounds[1][0], problem.bounds[1][1], resolution)
    best = problem.initial_point.copy()
    best_val = float("inf")
    for y in y_values:
        for x in x_values:
            point = np.array([x, y])
            if not problem.feasible(point):
                continue
            value = problem.objective(point)
            if value < best_val:
                best_val = value
                best = point
    return best
