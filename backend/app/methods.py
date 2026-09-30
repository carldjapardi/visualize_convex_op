from __future__ import annotations

import numpy as np

from .problem_definition import ProblemDefinition
from .problems import METHODS, get_problem, method_summary, sample_geometry
from .schemas import Iterate, MethodSummary, SolveRequest, SolveResponse
from .solvers import solve_first_order, solve_interior_point, solve_simplex
from .solvers.result import SolveResult


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
        result = solve_simplex(problem)
    elif method == "interior_point":
        result = solve_interior_point(problem)
    else:
        result = solve_first_order(problem, request)

    return _to_response(problem, method, result)


def _to_response(
    problem: ProblemDefinition,
    method_id: str,
    result: SolveResult,
) -> SolveResponse:
    trace = result.trace
    if trace and problem.type in {"lp", "qp", "qcqp", "socp"} and len(trace) == 1:
        x = np.array(trace[-1].x)
        trace = [
            trace[0],
            Iterate(
                iteration=1,
                x=trace[0].x,
                objective=trace[0].objective,
                violation=trace[0].violation,
                step_norm=float(np.linalg.norm(x - problem.initial_point)),
            ),
        ]

    return SolveResponse(
        problem=problem.summary(),
        method=method_summary(method_id),  # type: ignore[arg-type]
        status=result.status,
        compatible=True,
        solver=result.solver,
        message=result.message,
        objective_value=result.objective_value,
        variables=result.variables,
        iterations=result.iterations,
        trace=trace,
        history=result.history,
        geometry=sample_geometry(problem.id),
    )
