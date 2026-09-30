from __future__ import annotations

import numpy as np

from ..problem_definition import ProblemDefinition
from .helpers import circle_boundary


SOCP_RHS_COEFFICIENTS = {
    "socp-cone": (1.85, -0.25, 0.1),
    "socp-robust-line": (1.2, -0.15, 0.2),
    "socp-hyperbolic": (0.5, 0.75, 0.0),
    "socp-norm-ball-intersect": (2.0, 0.0, 0.0),
    "socp-portfolio-risk": (1.5, -0.3, 0.15),
}


def socp_rhs(x: np.ndarray, problem_id: str) -> float:
    base, coef_x, coef_y = SOCP_RHS_COEFFICIENTS[problem_id]
    return float(base + coef_x * x[0] + coef_y * x[1])


def socp_rhs_gradient(problem_id: str) -> np.ndarray:
    _, coef_x, coef_y = SOCP_RHS_COEFFICIENTS[problem_id]
    return np.array([coef_x, coef_y])


def _make_socp(
    problem_id: str,
    name: str,
    description: str,
    constraints: list[str],
    bounds: tuple[tuple[float, float], tuple[float, float]],
    objective_coef: np.ndarray,
    initial_point: np.ndarray,
    boundary_fn,
) -> ProblemDefinition:
    def feasible(x: np.ndarray) -> bool:
        return bool(np.linalg.norm(x) <= socp_rhs(x, problem_id) + 1e-9)

    def violation(x: np.ndarray) -> float:
        return float(max(0.0, np.linalg.norm(x) - socp_rhs(x, problem_id)))

    return ProblemDefinition(
        id=problem_id,
        type="socp",
        name=name,
        description=description,
        dimension=2,
        variables=["x", "y"],
        compatible_methods=["interior_point"],
        constraints=constraints,
        bounds=bounds,
        initial_point=initial_point,
        objective=lambda x: float(objective_coef @ x),
        gradient=lambda _: objective_coef.copy(),
        hessian=lambda _: np.zeros((2, 2)),
        feasible=feasible,
        violation=violation,
        boundaries_fn=boundary_fn,
        annotations=["The domain is a two-variable section of a second-order cone constraint."],
    )


def build_socp_problems() -> dict[str, ProblemDefinition]:
    return {
        "socp-cone": _make_socp(
            "socp-cone",
            "SOCP: Tilted Norm Bound",
            "A linear objective under a two-dimensional norm bound with an affine budget.",
            ["||(x, y)||_2 <= 1.85 - 0.25x + 0.1y"],
            ((-2.2, 2.2), (-2.2, 2.2)),
            np.array([-1.0, -0.7]),
            np.array([0.0, 0.0]),
            lambda: [],
        ),
        "socp-robust-line": _make_socp(
            "socp-robust-line",
            "SOCP: Affine Norm Budget",
            "A tighter affine budget changes the shape of the norm-feasible region.",
            ["||(x, y)||_2 <= 1.2 - 0.15x + 0.2y"],
            ((-1.8, 1.8), (-1.8, 1.8)),
            np.array([-0.8, -1.1]),
            np.array([0.1, 0.1]),
            lambda: [circle_boundary(1.2)],
        ),
        "socp-hyperbolic": _make_socp(
            "socp-hyperbolic",
            "SOCP: Steep Cone Section",
            "A norm constraint with a steep affine right-hand side.",
            ["||(x, y)||_2 <= 0.5 + 0.75x"],
            ((-0.5, 2.4), (-1.5, 1.5)),
            np.array([-1.2, -0.4]),
            np.array([0.0, 0.0]),
            lambda: [],
        ),
        "socp-norm-ball-intersect": ProblemDefinition(
            id="socp-norm-ball-intersect",
            type="socp",
            name="SOCP: Norm Ball and Box",
            description="Second-order cone intersected with a box in the plane.",
            dimension=2,
            variables=["x", "y"],
            compatible_methods=["interior_point"],
            constraints=["||(x, y)||_2 <= 2", "x >= -1", "y >= -1", "x <= 1.5", "y <= 1.5"],
            bounds=((-1.5, 2.0), (-1.5, 2.0)),
            initial_point=np.array([0.2, 0.2]),
            objective=lambda x: float(-1.0 * x[0] - 0.5 * x[1]),
            gradient=lambda _: np.array([-1.0, -0.5]),
            hessian=lambda _: np.zeros((2, 2)),
            feasible=lambda x: bool(
                np.linalg.norm(x) <= 2.0 + 1e-9
                and -1.0 <= x[0] <= 1.5 + 1e-9
                and -1.0 <= x[1] <= 1.5 + 1e-9
            ),
            violation=lambda x: float(
                max(0.0, np.linalg.norm(x) - 2.0, -1.0 - x[0], x[0] - 1.5, -1.0 - x[1], x[1] - 1.5)
            ),
            boundaries_fn=lambda: [circle_boundary(2.0)],
            annotations=["The domain is a two-variable section of a second-order cone constraint."],
        ),
        "socp-portfolio-risk": _make_socp(
            "socp-portfolio-risk",
            "SOCP: Linear Reward and Norm Budget",
            "A linear reward objective pushes against a second-order norm budget.",
            ["||(x, y)||_2 <= 1.5 - 0.3x + 0.15y"],
            ((-1.5, 2.0), (-1.5, 2.0)),
            np.array([-1.5, -0.9]),
            np.array([0.0, 0.0]),
            lambda: [],
        ),
    }
