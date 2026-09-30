from __future__ import annotations

import numpy as np

from ..problem_definition import LPData, ProblemDefinition
from .simplex_2d import _enumerate_vertices
from .result import SolveResult
from .utils import grid_search_minimize, iterate_point, round_matrix, round_vector, safe_solve


def solve_interior_point(problem: ProblemDefinition) -> SolveResult:
    if problem.type == "lp":
        return _solve_lp_barrier(problem)
    if problem.type == "qp":
        return _solve_qp_kkt(problem)
    if problem.type == "qcqp":
        return _solve_qcqp_barrier(problem)
    if problem.type == "socp":
        return _solve_socp_barrier(problem)
    if problem.type == "sdp":
        return _solve_sdp_barrier(problem)
    return SolveResult(status="failed", solver="Educational interior point", message="Unsupported type.")


def _solve_lp_barrier(problem: ProblemDefinition) -> SolveResult:
    if problem.lp_data is None:
        return SolveResult(status="failed", solver="Educational log-barrier IPM", message="Missing LP data.")

    lp = problem.lp_data
    x = problem.initial_point.astype(float).copy()
    if np.any(_lp_slacks(x, lp) <= 0):
        return SolveResult(status="failed", solver="Educational log-barrier LP", message="A strictly feasible starting point is required.")

    trace = [iterate_point(problem, 0, x, 0.0)]
    mu = 1.0
    iteration = 0

    for _outer in range(12):
        for _ in range(25):
            grad = _lp_barrier_gradient(x, lp, mu)
            hess = _lp_barrier_hessian(x, lp, mu)
            step = safe_solve(hess, grad)
            alpha = 1.0
            current_value = _lp_barrier_value(x, lp, mu)
            while alpha > 1e-10:
                x_new = x - alpha * step
                if np.all(_lp_slacks(x_new, lp) > 0) and _lp_barrier_value(x_new, lp, mu) <= current_value - 1e-4 * alpha * float(grad @ step):
                    break
                alpha *= 0.5
            if alpha <= 1e-10:
                break
            step_norm = float(np.linalg.norm(x_new - x))
            x = x_new
            iteration += 1
            trace.append(iterate_point(problem, iteration, x, step_norm))
            if step_norm < 1e-7:
                break
        mu *= 0.25
        if mu < 1e-7:
            break

    vertices = _enumerate_vertices(lp)
    if not vertices:
        return SolveResult(status="failed", solver="Educational log-barrier LP", message="No feasible LP vertex found.", trace=trace)
    optimal = min(vertices, key=problem.objective)
    if np.linalg.norm(optimal - x) > 1e-6:
        trace.append(iterate_point(problem, len(trace), optimal, float(np.linalg.norm(optimal - x))))
    x = optimal

    return SolveResult(
        status="optimal",
        solver="Educational log-barrier LP",
        message="Feasible barrier path followed by an exact LP corner refinement.",
        objective_value=problem.objective(x),
        variables={"x": round_vector(x)},
        iterations=len(trace) - 1,
        trace=trace,
        history={"objective": [point.objective for point in trace], "violation": [point.violation for point in trace]},
    )


def _solve_qp_kkt(problem: ProblemDefinition) -> SolveResult:
    x = problem.initial_point.astype(float).copy()
    trace = [iterate_point(problem, 0, x, 0.0)]

    for iteration in range(1, 80):
        if problem.gradient is None or problem.hessian is None:
            break
        grad = problem.gradient(x)
        hess = problem.hessian(x)
        step = safe_solve(hess, grad)
        x_new = x - step
        if problem.project is not None:
            x_new = problem.project(x_new)
        step_norm = float(np.linalg.norm(x_new - x))
        x = x_new
        trace.append(iterate_point(problem, iteration, x, step_norm))
        if step_norm < 1e-8:
            break

    refined = grid_search_minimize(problem, resolution=100)
    if problem.objective(refined) < problem.objective(x) and problem.feasible(refined):
        trace.append(iterate_point(problem, len(trace), refined, float(np.linalg.norm(refined - x))))
        x = refined

    return SolveResult(
        status="approximate",
        solver="Projected Newton QP",
        message="Projected Newton steps, with a feasible grid refinement when it improves the objective.",
        objective_value=problem.objective(x),
        variables={"x": round_vector(x), "constraint_violation": problem.violation(x)},
        iterations=len(trace) - 1,
        trace=trace,
        history={"objective": [point.objective for point in trace]},
    )


def _solve_qcqp_barrier(problem: ProblemDefinition) -> SolveResult:
    x = problem.initial_point.astype(float).copy()
    if problem.project is not None:
        x = problem.project(x)
    trace = [iterate_point(problem, 0, x, 0.0)]
    mu = 1.0
    iteration = 0

    for _outer in range(10):
        for _ in range(30):
            if problem.gradient is None:
                break
            grad = problem.gradient(x) + _violation_barrier_grad(x, problem, mu)
            x_new = x - 0.08 * grad
            if problem.project is not None:
                x_new = problem.project(x_new)
            step_norm = float(np.linalg.norm(x_new - x))
            x = x_new
            iteration += 1
            trace.append(iterate_point(problem, iteration, x, step_norm))
            if step_norm < 1e-7:
                break
        mu *= 0.25

    refined = grid_search_minimize(problem, resolution=100)
    if problem.objective(refined) < problem.objective(x) and problem.feasible(refined):
        trace.append(iterate_point(problem, len(trace), refined, float(np.linalg.norm(refined - x))))
        x = refined

    return SolveResult(
        status="approximate",
        solver="Projected QCQP reference",
        message="Projected gradient steps with a constraint-violation penalty and feasible grid refinement.",
        objective_value=problem.objective(x),
        variables={"x": round_vector(x), "constraint_violation": problem.violation(x)},
        iterations=len(trace) - 1,
        trace=trace,
        history={"objective": [point.objective for point in trace]},
    )


def _violation_barrier_grad(x: np.ndarray, problem: ProblemDefinition, mu: float) -> np.ndarray:
    grad = np.zeros(2)
    eps = 1e-5
    base = problem.violation(x)
    for i in range(2):
        xp = x.copy()
        xp[i] += eps
        grad[i] = mu * (problem.violation(xp) - base) / eps
    return grad


def _solve_socp_barrier(problem: ProblemDefinition) -> SolveResult:
    x = np.zeros(2)
    trace = [iterate_point(problem, 0, x, 0.0)]
    mu = 1.0

    for outer in range(1, 10):
        for inner in range(25):
            grad = problem.gradient(x) if problem.gradient else np.zeros(2)
            slack = _socp_slack(x, problem)
            if slack > 1e-8:
                grad = grad + (mu / slack) * _socp_barrier_grad(x, problem)
            delta = -0.15 * grad
            delta_norm = np.linalg.norm(delta)
            if delta_norm > 0.25:
                delta *= 0.25 / delta_norm
            x_new = x + delta
            x_new = _project_socp(x_new, problem)
            step_norm = float(np.linalg.norm(x_new - x))
            x = x_new
            trace.append(iterate_point(problem, len(trace), x, step_norm))
            if step_norm < 1e-7:
                break
        mu *= 0.2

    refined = grid_search_minimize(problem, resolution=100)
    if problem.objective(refined) < problem.objective(x) and problem.feasible(refined):
        trace.append(iterate_point(problem, len(trace), refined, float(np.linalg.norm(refined - x))))
        x = refined

    return SolveResult(
        status="approximate",
        solver="Projected SOCP reference",
        message="Projected updates for a norm constraint, with feasible grid refinement.",
        objective_value=problem.objective(x),
        variables={"x": round_vector(x), "constraint_violation": problem.violation(x)},
        iterations=len(trace) - 1,
        trace=trace,
        history={"objective": [point.objective for point in trace]},
    )


def _socp_slack(x: np.ndarray, problem: ProblemDefinition) -> float:
    from ..problem_catalog.socp import socp_rhs

    return socp_rhs(x, problem.id) - np.linalg.norm(x)


def _socp_barrier_grad(x: np.ndarray, problem: ProblemDefinition) -> np.ndarray:
    from ..problem_catalog.socp import socp_rhs_gradient

    norm = np.linalg.norm(x)
    direction = x / norm if norm > 1e-8 else np.zeros(2)
    return direction - socp_rhs_gradient(problem.id)


def _project_socp(x: np.ndarray, problem: ProblemDefinition) -> np.ndarray:
    point = x.astype(float).copy()
    if problem.id == "socp-norm-ball-intersect":
        point = np.clip(point, [-1.0, -1.0], [1.5, 1.5])
    if problem.feasible(point):
        return point
    low, high = 0.0, 1.0
    for _ in range(50):
        middle = (low + high) / 2.0
        if problem.feasible(middle * point):
            low = middle
        else:
            high = middle
    return (0.999 * low) * point


def _solve_sdp_barrier(problem: ProblemDefinition) -> SolveResult:
    a, b = problem.initial_point.astype(float)
    x = np.array([a, b])
    if not problem.feasible(x):
        x = np.array([0.5, 0.0])
    trace = [iterate_point(problem, 0, x, 0.0)]
    mu = 1.0

    for _outer in range(12):
        for _inner in range(30):
            grad = problem.gradient(x) if problem.gradient else np.zeros(2)
            a_val, b_val = x
            psd_viol = b_val * b_val - a_val * (1.0 - a_val)
            if psd_viol > 1e-9:
                grad[0] -= mu * (1.0 - 2 * a_val)
                grad[1] -= mu * (2 * b_val)
            x_new = x - 0.1 * grad
            x_new[0] = np.clip(x_new[0], 0.01, 0.99)
            x_new[1] = np.clip(x_new[1], -np.sqrt(x_new[0] * (1 - x_new[0])) * 0.98, np.sqrt(x_new[0] * (1 - x_new[0])) * 0.98)
            step_norm = float(np.linalg.norm(x_new - x))
            x = x_new
            trace.append(iterate_point(problem, len(trace), x, step_norm))
            if step_norm < 1e-7:
                break
        mu *= 0.2

    refined = grid_search_minimize(problem, resolution=100)
    if problem.objective(refined) < problem.objective(x) and problem.feasible(refined):
        trace.append(iterate_point(problem, len(trace), refined, float(np.linalg.norm(refined - x))))
        x = refined

    matrix = np.array([[x[0], x[1]], [x[1], 1.0 - x[0]]])
    eigenvalues = np.linalg.eigvalsh(matrix)

    return SolveResult(
        status="approximate",
        solver="Projected PSD-slice reference",
        message="Projected updates on a trace-one PSD slice, with eigenvalue diagnostics and feasible grid refinement.",
        objective_value=problem.objective(x),
        variables={
            "X": round_matrix(matrix),
            "plot_point": round_vector(x),
            "eigenvalues": round_vector(eigenvalues),
            "trace": float(np.trace(matrix)),
        },
        iterations=len(trace) - 1,
        trace=trace,
        history={"objective": [point.objective for point in trace], "eigenvalues": [float(value) for value in eigenvalues]},
    )


def _lp_barrier_gradient(x: np.ndarray, lp, mu: float) -> np.ndarray:
    grad = lp.c.copy()
    slacks = lp.b_ub - lp.a_ub @ x
    for row, slack in zip(lp.a_ub, slacks):
        if slack > 1e-8:
            grad += (mu / slack) * row
    (x_low, x_high), (y_low, y_high) = lp.bounds
    if x_low is not None and x[0] - x_low > 1e-8:
        grad[0] -= mu / (x[0] - x_low)
    if x_high is not None and x_high - x[0] > 1e-8:
        grad[0] += mu / (x_high - x[0])
    if y_low is not None and x[1] - y_low > 1e-8:
        grad[1] -= mu / (x[1] - y_low)
    if y_high is not None and y_high - x[1] > 1e-8:
        grad[1] += mu / (y_high - x[1])
    return grad


def _lp_barrier_hessian(x: np.ndarray, lp, mu: float) -> np.ndarray:
    hess = np.zeros((2, 2))
    slacks = lp.b_ub - lp.a_ub @ x
    for row, slack in zip(lp.a_ub, slacks):
        if slack > 1e-8:
            outer = np.outer(row, row)
            hess += (mu / slack**2) * outer
    (x_low, x_high), (y_low, y_high) = lp.bounds
    if x_low is not None and x[0] - x_low > 1e-8:
        hess[0, 0] += mu / (x[0] - x_low) ** 2
    if x_high is not None and x_high - x[0] > 1e-8:
        hess[0, 0] += mu / (x_high - x[0]) ** 2
    if y_low is not None and x[1] - y_low > 1e-8:
        hess[1, 1] += mu / (x[1] - y_low) ** 2
    if y_high is not None and y_high - x[1] > 1e-8:
        hess[1, 1] += mu / (y_high - x[1]) ** 2
    return hess + 1e-8 * np.eye(2)


def _lp_slacks(x: np.ndarray, lp: LPData) -> np.ndarray:
    slacks = list(lp.b_ub - lp.a_ub @ x)
    for coordinate, (lower, upper) in zip(x, lp.bounds):
        if lower is not None:
            slacks.append(coordinate - lower)
        if upper is not None:
            slacks.append(upper - coordinate)
    return np.asarray(slacks, dtype=float)


def _lp_barrier_value(x: np.ndarray, lp: LPData, mu: float) -> float:
    return float(lp.c @ x - mu * np.log(_lp_slacks(x, lp)).sum())
