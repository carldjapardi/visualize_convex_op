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
    current = problem.initial_point.astype(float).copy()
    if method == "pgd" and problem.project is not None:
        current = problem.project(current)
    trace = [iterate_point(problem, 0, current, 0.0)]

    for iteration in range(1, request.max_iterations + 1):
        gradient = problem.gradient(current)
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

        step_norm = float(np.linalg.norm(next_point - current))
        current = next_point
        trace.append(iterate_point(problem, iteration, current, step_norm))
        if step_norm < 1e-6:
            break

    best_point = min(trace, key=lambda item: item.objective + 1000.0 * item.violation)
    status = "completed" if all(point.violation <= 1e-5 for point in trace) else "completed_with_constraint_violation"

    message = problem.trace_message or _default_trace_message(method, status)

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
