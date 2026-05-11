from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np
from scipy.optimize import linprog

from .problems import (
    METHODS,
    ProblemDefinition,
    build_cvxpy_problem,
    get_problem,
    method_summary,
    sample_geometry,
    scipy_lp_data,
)
from .schemas import Iterate, MethodId, MethodSummary, SolveRequest, SolveResponse


def list_method_summaries() -> list[MethodSummary]:
    return [method_summary(method_id) for method_id in METHODS]


def solve(request: SolveRequest) -> SolveResponse:
    problem = get_problem(request.problem_id)
    method = request.method

    if method not in problem.compatible_methods:
        return SolveResponse(
            problem=problem.summary(),
            method=method_summary(method),
            status="unsupported",
            compatible=False,
            message=f"{method_summary(method).name} is not a meaningful method for {problem.name} in this demo.",
            geometry=sample_geometry(problem.id),
        )

    if method == "simplex":
        return _solve_lp_with_scipy(problem, method="highs-ds", method_id=method)
    if method == "interior_point":
        if problem.type == "lp":
            return _solve_lp_with_scipy(problem, method="highs-ipm", method_id=method)
        return _solve_with_cvxpy(problem, method_id=method)

    return _solve_trace_method(problem, request)


def _solve_lp_with_scipy(problem: ProblemDefinition, method: str, method_id: MethodId) -> SolveResponse:
    c, a_ub, b_ub, bounds = scipy_lp_data(problem)
    result = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method=method)
    variables: dict[str, Any] = {}
    trace: list[Iterate] = []

    if result.x is not None:
        x = np.asarray(result.x, dtype=float)
        variables = {"x": _round_vector(x), "objective_direction": _round_vector(c)}
        trace = _lp_vertex_trace(problem, x)

    return SolveResponse(
        problem=problem.summary(),
        method=method_summary(method_id),
        status="optimal" if result.success else "failed",
        compatible=True,
        solver=f"SciPy linprog({method})",
        message=str(result.message),
        objective_value=float(result.fun) if result.fun is not None else None,
        variables=variables,
        iterations=int(result.nit) if result.nit is not None else 0,
        trace=trace,
        history={"objective": [point.objective for point in trace]},
        geometry=sample_geometry(problem.id),
    )


def _solve_with_cvxpy(problem: ProblemDefinition, method_id: MethodId) -> SolveResponse:
    cvx_problem, variables = build_cvxpy_problem(problem)
    solver_name = _choose_solver(problem)
    message = "Solved with CVXPY."
    status = "failed"

    try:
        cvx_problem.solve(solver=solver_name, verbose=False)
    except Exception as exc:
        fallback = _fallback_solver(solver_name)
        try:
            cvx_problem.solve(solver=fallback, verbose=False)
            solver_name = fallback
            message = f"Solved with CVXPY after falling back from {solver_name}: {exc}"
        except Exception as fallback_exc:
            return SolveResponse(
                problem=problem.summary(),
                method=method_summary(method_id),
                status="failed",
                compatible=True,
                solver=f"CVXPY ({solver_name})",
                message=f"CVXPY failed: {fallback_exc}",
                geometry=sample_geometry(problem.id),
            )

    status = cvx_problem.status or status
    solution = _extract_variables(problem, variables)
    trace = _solution_trace(problem, solution)

    return SolveResponse(
        problem=problem.summary(),
        method=method_summary(method_id),
        status=status,
        compatible=True,
        solver=f"CVXPY ({solver_name})",
        message=message,
        objective_value=float(cvx_problem.value) if cvx_problem.value is not None else None,
        variables=solution,
        iterations=int(cvx_problem.solver_stats.num_iters or 0),
        trace=trace,
        history=_cvxpy_history(problem, solution),
        geometry=sample_geometry(problem.id),
    )


def _solve_trace_method(problem: ProblemDefinition, request: SolveRequest) -> SolveResponse:
    if problem.gradient is None:
        raise ValueError(f"{problem.id} has no gradient implementation")

    rng = np.random.default_rng(request.seed)
    current = problem.initial_point.astype(float).copy()
    trace: list[Iterate] = [_iterate(problem, 0, current, step_norm=0.0)]

    for iteration in range(1, request.max_iterations + 1):
        gradient = problem.gradient(current)
        if request.method == "sgd":
            gradient = gradient + rng.normal(scale=0.08 * max(1.0, np.linalg.norm(gradient)), size=gradient.shape)
        if request.method == "newton" and problem.hessian is not None:
            hessian = problem.hessian(current)
            step = _safe_solve(hessian, gradient)
        else:
            step = gradient

        next_point = current - request.step_size * step
        if request.method == "pgd" and problem.project is not None:
            next_point = problem.project(next_point)

        step_norm = float(np.linalg.norm(next_point - current))
        current = next_point
        trace.append(_iterate(problem, iteration, current, step_norm=step_norm))
        if step_norm < 1e-6:
            break

    best_point = min(trace, key=lambda item: item.objective + 1000.0 * item.violation)
    return SolveResponse(
        problem=problem.summary(),
        method=method_summary(request.method),
        status="completed" if best_point.violation <= 1e-5 else "completed_with_constraint_violation",
        compatible=True,
        solver="Backend trace implementation",
        message="Generated an algorithm trajectory; use Interior Point for a certified solver reference.",
        objective_value=best_point.objective,
        variables={"x": _round_vector(np.array(best_point.x)), "best_iteration": best_point.iteration},
        iterations=len(trace) - 1,
        trace=trace,
        history={
            "objective": [point.objective for point in trace],
            "violation": [point.violation for point in trace],
            "step_norm": [point.step_norm for point in trace],
        },
        geometry=sample_geometry(problem.id),
    )


def _choose_solver(problem: ProblemDefinition) -> str:
    installed = set(cp.installed_solvers())
    if problem.type == "qp" and "OSQP" in installed:
        return "OSQP"
    if problem.type in {"socp", "sdp", "qcqp"} and "CLARABEL" in installed:
        return "CLARABEL"
    if "SCS" in installed:
        return "SCS"
    if installed:
        return sorted(installed)[0]
    return "SCS"


def _fallback_solver(previous: str) -> str:
    installed = [solver for solver in cp.installed_solvers() if solver != previous]
    if "SCS" in installed:
        return "SCS"
    if "CLARABEL" in installed:
        return "CLARABEL"
    return installed[0] if installed else previous


def _extract_variables(problem: ProblemDefinition, variables: dict[str, Any]) -> dict[str, Any]:
    if problem.type == "sdp":
        matrix = np.asarray(variables["X"].value, dtype=float)
        eigenvalues = np.linalg.eigvalsh(matrix)
        return {
            "X": _round_matrix(matrix),
            "plot_point": _round_vector(np.array([matrix[0, 0], matrix[0, 1]])),
            "eigenvalues": _round_vector(eigenvalues),
            "trace": float(np.trace(matrix)),
        }

    vector = np.asarray(variables["x"].value, dtype=float)
    return {"x": _round_vector(vector), "constraint_violation": problem.violation(vector)}


def _solution_trace(problem: ProblemDefinition, solution: dict[str, Any]) -> list[Iterate]:
    if problem.type == "sdp":
        x = np.asarray(solution["plot_point"], dtype=float)
    else:
        x = np.asarray(solution["x"], dtype=float)

    return [
        _iterate(problem, 0, problem.initial_point, step_norm=0.0),
        _iterate(problem, 1, x, step_norm=float(np.linalg.norm(x - problem.initial_point))),
    ]


def _cvxpy_history(problem: ProblemDefinition, solution: dict[str, Any]) -> dict[str, list[float]]:
    if problem.type == "sdp":
        return {"eigenvalues": [float(value) for value in solution.get("eigenvalues", [])]}
    if "x" in solution:
        x = np.asarray(solution["x"], dtype=float)
        return {"objective": [problem.objective(problem.initial_point), problem.objective(x)]}
    return {}


def _lp_vertex_trace(problem: ProblemDefinition, solution: np.ndarray) -> list[Iterate]:
    if problem.id == "lp-warehouse":
        vertices = [
            np.array([0.0, 0.0]),
            np.array([3.5, 0.0]),
            np.array([3.5, 1.0]),
            np.array([3.0, 2.0]),
            np.array([0.0, 3.0]),
        ]
    else:
        vertices = [
            np.array([0.0, 0.0]),
            np.array([3.0, 0.0]),
            np.array([3.0, 1.0]),
            np.array([2.0, 2.0]),
            np.array([0.0, 2.0]),
        ]
    ordered = sorted(vertices, key=problem.objective, reverse=True)
    if not any(np.allclose(solution, point) for point in ordered):
        ordered.append(solution)
    else:
        ordered = [point for point in ordered if problem.objective(point) >= problem.objective(solution)] + [solution]
    return [_iterate(problem, idx, point, step_norm=0.0 if idx == 0 else float(np.linalg.norm(point - ordered[idx - 1]))) for idx, point in enumerate(ordered)]


def _iterate(problem: ProblemDefinition, iteration: int, x: np.ndarray, step_norm: float) -> Iterate:
    return Iterate(
        iteration=iteration,
        x=_round_vector(x),
        objective=round(problem.objective(x), 8),
        violation=round(problem.violation(x), 8),
        step_norm=round(step_norm, 8),
    )


def _safe_solve(hessian: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    regularized = hessian + 1e-8 * np.eye(hessian.shape[0])
    try:
        return np.linalg.solve(regularized, gradient)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(regularized) @ gradient


def _round_vector(vector: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in np.ravel(vector)]


def _round_matrix(matrix: np.ndarray) -> list[list[float]]:
    return [[round(float(value), 6) for value in row] for row in matrix]
