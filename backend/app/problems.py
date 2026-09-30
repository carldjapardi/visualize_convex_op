from __future__ import annotations

from typing import Any

import numpy as np

from .problem_definition import ProblemDefinition
from .problem_catalog.catalog import build_all_problems
from .schemas import GeometryResponse, MethodId, MethodSummary, ProblemSummary, ProblemType

PROBLEMS: dict[str, ProblemDefinition] = build_all_problems()

METHODS: dict[MethodId, dict[str, object]] = {
    "simplex": {
        "name": "Simplex",
        "description": "Educational 2D vertex-to-vertex simplex for linear programs.",
        "best_for": ["lp"],
    },
    "interior_point": {
        "name": "Constrained Reference",
        "description": "A feasible reference path: LP log barrier, or projected steps with sampled refinement for other classes.",
        "best_for": ["lp", "qp", "qcqp", "socp", "sdp"],
    },
    "gd": {
        "name": "Gradient Descent",
        "description": "Educational first-order descent trace for smooth objectives.",
        "best_for": ["qp", "qcqp", "nonconvex"],
    },
    "sgd": {
        "name": "Stochastic Gradient Descent",
        "description": "Noisy first-order descent trace for smooth objectives.",
        "best_for": ["qp", "qcqp", "nonconvex"],
    },
    "pgd": {
        "name": "Projected Gradient Descent",
        "description": "Gradient step followed by projection onto the feasible set.",
        "best_for": ["qp", "qcqp"],
    },
    "newton": {
        "name": "Newton's Method",
        "description": "Second-order educational trace using the local Hessian.",
        "best_for": ["qp", "qcqp", "nonconvex"],
    },
}


def method_summary(method_id: MethodId) -> MethodSummary:
    item = METHODS[method_id]
    return MethodSummary(
        id=method_id,
        name=str(item["name"]),
        description=str(item["description"]),
        best_for=item["best_for"],  # type: ignore[arg-type]
    )


def list_problem_summaries() -> list[ProblemSummary]:
    return [problem.summary() for problem in PROBLEMS.values()]


def get_problem(problem_id: str) -> ProblemDefinition:
    if problem_id not in PROBLEMS:
        raise KeyError(f"Unknown problem id: {problem_id}")
    return PROBLEMS[problem_id]


def sample_geometry(problem_id: str, resolution: int = 80) -> GeometryResponse:
    problem = get_problem(problem_id)
    x_values = np.linspace(problem.bounds[0][0], problem.bounds[0][1], resolution)
    y_values = np.linspace(problem.bounds[1][0], problem.bounds[1][1], resolution)
    objective: list[list[float | None]] = []
    feasible: list[list[int]] = []

    for y in y_values:
        objective_row: list[float | None] = []
        feasible_row: list[int] = []
        for x in x_values:
            point = np.array([x, y])
            is_feasible = problem.feasible(point)
            feasible_row.append(1 if is_feasible else 0)
            value = problem.objective(point)
            objective_row.append(value if np.isfinite(value) else None)
        objective.append(objective_row)
        feasible.append(feasible_row)

    annotations = list(problem.annotations)
    if problem.type == "sdp" and not annotations:
        annotations.append("The domain is a trace-one slice of 2×2 PSD matrices; 3D height is objective value.")
    if problem.type == "socp" and not annotations:
        annotations.append("The domain is a two-variable section of a second-order cone constraint.")

    return GeometryResponse(
        problem_id=problem.id,
        problem_type=problem.type,
        x=x_values.round(5).tolist(),
        y=y_values.round(5).tolist(),
        objective=objective,
        feasible=feasible,
        boundaries=problem.boundaries(),
        annotations=annotations,
    )
