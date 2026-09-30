from __future__ import annotations

import itertools

import numpy as np

from ..problem_definition import LPData, ProblemDefinition
from .result import SolveResult
from .utils import iterate_point, round_vector


def solve_simplex(problem: ProblemDefinition) -> SolveResult:
    if problem.lp_data is None:
        return SolveResult(
            status="failed",
            solver="Educational 2D simplex",
            message="No LP encoding available for this problem.",
        )

    vertices = _enumerate_vertices(problem.lp_data)
    if not vertices:
        return SolveResult(
            status="failed",
            solver="Educational 2D simplex",
            message="No feasible vertices found.",
        )

    start = max(vertices, key=problem.objective)
    trace = [iterate_point(problem, 0, start, 0.0)]
    current = start
    visited = {tuple(np.round(current, 8))}

    for step in range(1, len(vertices) + 2):
        neighbors = _improving_neighbors(current, vertices, problem, visited)
        if not neighbors:
            break
        next_vertex = min(neighbors, key=problem.objective)
        step_norm = float(np.linalg.norm(next_vertex - current))
        current = next_vertex
        visited.add(tuple(np.round(current, 8)))
        trace.append(iterate_point(problem, step, current, step_norm))

    optimal = min(vertices, key=problem.objective)
    if trace[-1].objective > problem.objective(optimal) + 1e-8:
        trace.append(
            iterate_point(
                problem,
                len(trace),
                optimal,
                float(np.linalg.norm(optimal - np.array(trace[-1].x))),
            )
        )

    return SolveResult(
        status="optimal",
        solver="Educational 2D simplex",
        message="Vertex-to-vertex simplex walk over LP feasible corners.",
        objective_value=problem.objective(optimal),
        variables={"x": round_vector(optimal)},
        iterations=max(0, len(trace) - 1),
        trace=trace,
        history={"objective": [point.objective for point in trace]},
    )


def _improving_neighbors(
    current: np.ndarray,
    vertices: list[np.ndarray],
    problem: ProblemDefinition,
    visited: set[tuple[float, ...]],
) -> list[np.ndarray]:
    current_obj = problem.objective(current)
    if problem.lp_data is None:
        return []
    lines = _constraint_lines(problem.lp_data)
    active = {index for index, (row, limit) in enumerate(lines) if abs(row @ current - limit) < 1e-6}
    candidates: list[np.ndarray] = []
    for vertex in vertices:
        key = tuple(np.round(vertex, 8))
        if key in visited:
            continue
        adjacent = any(abs(lines[index][0] @ vertex - lines[index][1]) < 1e-6 for index in active)
        if adjacent and problem.objective(vertex) < current_obj - 1e-9:
            candidates.append(vertex)
    return candidates


def _enumerate_vertices(lp: LPData) -> list[np.ndarray]:
    lines = _constraint_lines(lp)
    candidates: list[np.ndarray] = []
    for (a1, b1), (a2, b2) in itertools.combinations(lines, 2):
        point = _line_intersection(a1, b1, a2, b2)
        if point is None:
            continue
        if _is_feasible(point, lp):
            candidates.append(point)

    unique: list[np.ndarray] = []
    for point in candidates:
        if not any(np.allclose(point, existing, atol=1e-6) for existing in unique):
            unique.append(point)
    return unique


def _constraint_lines(lp: LPData) -> list[tuple[np.ndarray, float]]:
    lines: list[tuple[np.ndarray, float]] = []
    for row, limit in zip(lp.a_ub, lp.b_ub):
        lines.append((np.asarray(row, dtype=float), float(limit)))
    (x_low, x_high), (y_low, y_high) = lp.bounds
    if x_low is not None:
        lines.append((np.array([1.0, 0.0]), x_low))
    if x_high is not None:
        lines.append((np.array([-1.0, 0.0]), -x_high))
    if y_low is not None:
        lines.append((np.array([0.0, 1.0]), y_low))
    if y_high is not None:
        lines.append((np.array([0.0, -1.0]), -y_high))
    return lines


def _line_intersection(
    a1: np.ndarray,
    b1: float,
    a2: np.ndarray,
    b2: float,
) -> np.ndarray | None:
    matrix = np.vstack([a1, a2])
    rhs = np.array([b1, b2])
    if abs(np.linalg.det(matrix)) < 1e-10:
        return None
    return np.linalg.solve(matrix, rhs)


def _is_feasible(point: np.ndarray, lp: LPData) -> bool:
    if np.any(lp.a_ub @ point > lp.b_ub + 1e-8):
        return False
    (x_low, x_high), (y_low, y_high) = lp.bounds
    if x_low is not None and point[0] < x_low - 1e-8:
        return False
    if x_high is not None and point[0] > x_high + 1e-8:
        return False
    if y_low is not None and point[1] < y_low - 1e-8:
        return False
    if y_high is not None and point[1] > y_high + 1e-8:
        return False
    return True
