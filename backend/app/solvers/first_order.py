from __future__ import annotations

import numpy as np

from ..problem_definition import ProblemDefinition
from ..schemas import MethodId, SolveRequest
from .result import SolveResult
from .utils import iterate_point, round_vector, safe_solve


def solve_first_order(problem: ProblemDefinition, request: SolveRequest) -> SolveResult:
    if problem.gradient is None:
        raise ValueError(f"{problem.id} has no gradient implementation")

    method: MethodId = request.method
    rng = np.random.default_rng(request.seed)
    start = request.start_point if problem.type == "nonconvex" and request.start_point is not None else problem.initial_point
    current = np.array(start, dtype=float)
    if not np.all(np.isfinite(current)):
        raise ValueError("Starting coordinates must be finite")
    if method == "pgd" and problem.project is not None:
        current = problem.project(current)
    trace = [iterate_point(problem, 0, current, 0.0)]
    stopped_at_window = False

    for iteration in range(1, request.max_iterations + 1):
        gradient = problem.gradient(current)
        if not np.all(np.isfinite(gradient)):
            stopped_at_window = True
            break
        if method == "sgd":
            gradient = gradient + rng.normal(
                scale=0.08 * max(1.0, np.linalg.norm(gradient)),
                size=gradient.shape,
            )
        if method == "newton" and problem.hessian is not None:
            step = safe_solve(problem.hessian(current), gradient)
        else:
            step = gradient

        next_point = current - request.step_size * step
        if method == "pgd" and problem.project is not None:
            next_point = problem.project(next_point)
        if problem.type == "nonconvex" and (
            not np.all(np.isfinite(next_point))
            or any(next_point[i] < axis[0] or next_point[i] > axis[1] for i, axis in enumerate(problem.bounds))
        ):
            stopped_at_window = True
            break

        step_norm = float(np.linalg.norm(next_point - current))
        current = next_point
        trace.append(iterate_point(problem, iteration, current, step_norm))
        if step_norm < 1e-6:
            break

    best_point = min(trace, key=lambda item: item.objective + 1000.0 * item.violation)
    status = "left_plot_window" if stopped_at_window else (
        "completed" if all(point.violation <= 1e-5 for point in trace) else "completed_with_constraint_violation"
    )

    message = "The next step left the plotted window. Reduce the step size or choose another starting point." if stopped_at_window else problem.trace_message or (
        "Newton's local curvature can move uphill or toward a saddle on a nonconvex surface."
        if problem.type == "nonconvex" and method == "newton"
        else "Local descent path on a nonconvex surface; the starting basin can change the endpoint."
        if problem.type == "nonconvex"
        else _default_trace_message(method, status)
    )

    return SolveResult(
        status=status,
        solver="Educational first-order trace",
        message=message,
        objective_value=best_point.objective,
        variables={"x": round_vector(np.array(best_point.x)), "best_iteration": best_point.iteration, "constraint_violation": best_point.violation},
        iterations=len(trace) - 1,
        trace=trace,
        history={
            "objective": [point.objective for point in trace],
            "violation": [point.violation for point in trace],
            "step_norm": [point.step_norm for point in trace],
        },
    )


def _default_trace_message(method: MethodId, status: str) -> str:
    if status == "completed_with_constraint_violation" and method in {"gd", "sgd"}:
        return "GD/SGD stepped outside the feasible set — compare with projected gradient descent or the constrained reference."
    if method == "pgd":
        return "Projected gradient descent trajectory with feasibility projection."
    if method == "newton":
        return "Newton step trajectory using the local Hessian."
    return "First-order educational trajectory; compare it with the constrained reference path."
