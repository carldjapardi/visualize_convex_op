from __future__ import annotations

import numpy as np

from ..problem_definition import ProblemDefinition


def _double_well(x: np.ndarray) -> float:
    return float((x[0] ** 2 - 1) ** 2 + 0.35 * x[1] ** 2 + 0.25 * x[0])


def _double_well_gradient(x: np.ndarray) -> np.ndarray:
    return np.array([4 * x[0] * (x[0] ** 2 - 1) + 0.25, 0.7 * x[1]])


def _double_well_hessian(x: np.ndarray) -> np.ndarray:
    return np.diag([12 * x[0] ** 2 - 4, 0.7])


def _himmelblau(x: np.ndarray) -> float:
    a = x[0] ** 2 + x[1] - 11
    b = x[0] + x[1] ** 2 - 7
    return float(a * a + b * b)


def _himmelblau_gradient(x: np.ndarray) -> np.ndarray:
    a = x[0] ** 2 + x[1] - 11
    b = x[0] + x[1] ** 2 - 7
    return np.array([4 * x[0] * a + 2 * b, 2 * a + 4 * x[1] * b])


def _himmelblau_hessian(x: np.ndarray) -> np.ndarray:
    a = x[0] ** 2 + x[1] - 11
    b = x[0] + x[1] ** 2 - 7
    return np.array([[4 * a + 8 * x[0] ** 2 + 2, 4 * (x[0] + x[1])],
                     [4 * (x[0] + x[1]), 4 * b + 8 * x[1] ** 2 + 2]])


def _rippled_bowl(x: np.ndarray) -> float:
    return float(0.18 * (x[0] ** 2 + x[1] ** 2) + 2 - np.cos(3 * x[0]) - np.cos(3 * x[1]) + 0.12 * x[0])


def _rippled_gradient(x: np.ndarray) -> np.ndarray:
    return np.array([0.36 * x[0] + 3 * np.sin(3 * x[0]) + 0.12,
                     0.36 * x[1] + 3 * np.sin(3 * x[1])])


def _rippled_hessian(x: np.ndarray) -> np.ndarray:
    return np.diag([0.36 + 9 * np.cos(3 * x[0]), 0.36 + 9 * np.cos(3 * x[1])])


def build_nonconvex_problems() -> dict[str, ProblemDefinition]:
    shared = {
        "type": "nonconvex",
        "dimension": 2,
        "variables": ["x", "y"],
        "compatible_methods": ["gd", "sgd", "newton"],
        "constraints": ["No constraints; the plot is a finite window"],
        "feasible": lambda _: True,
        "violation": lambda _: 0.0,
    }
    return {
        "nonconvex-double-well": ProblemDefinition(
            id="nonconvex-double-well", name="Nonconvex: Tilted Double Well",
            description="Two valleys have different depths. A local method can settle in the shallower one.",
            bounds=((-2.2, 2.2), (-1.8, 1.8)), initial_point=np.array([0.15, 1.15]),
            objective=_double_well, gradient=_double_well_gradient, hessian=_double_well_hessian,
            default_step_size=0.08, default_max_iterations=90,
            annotations=["Move the starting x coordinate across zero to compare the two basins."],
            **shared,
        ),
        "nonconvex-himmelblau": ProblemDefinition(
            id="nonconvex-himmelblau", name="Nonconvex: Four Basins",
            description="Himmelblau's function has four separated minima in the plotted window.",
            bounds=((-5.0, 5.0), (-5.0, 5.0)), initial_point=np.array([0.0, 0.0]),
            objective=_himmelblau, gradient=_himmelblau_gradient, hessian=_himmelblau_hessian,
            default_step_size=0.01, default_max_iterations=100,
            annotations=["Try starts in different quadrants; the destination depends on the basin."],
            **shared,
        ),
        "nonconvex-rippled-bowl": ProblemDefinition(
            id="nonconvex-rippled-bowl", name="Nonconvex: Rippled Bowl",
            description="A broad bowl contains many small local valleys created by oscillations.",
            bounds=((-3.2, 3.2), (-3.2, 3.2)), initial_point=np.array([2.4, -1.8]),
            objective=_rippled_bowl, gradient=_rippled_gradient, hessian=_rippled_hessian,
            default_step_size=0.06, default_max_iterations=100,
            annotations=["Move the start or change the step size to see which ripple traps the path."],
            **shared,
        ),
    }
