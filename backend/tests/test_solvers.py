from __future__ import annotations

from app.methods import solve
from app.schemas import SolveRequest
from app.solvers.utils import grid_search_minimize
from app.problems import get_problem
from app.problems import PROBLEMS

import numpy as np


def test_interior_point_lp():
    response = solve(
        SolveRequest(problem_id="lp-basic", method="interior_point", max_iterations=10, step_size=0.1, seed=0)
    )
    assert response.status == "optimal"
    assert "Educational" in (response.solver or "")


def test_pgd_qcqp_feasible():
    response = solve(
        SolveRequest(
            problem_id="qcqp-intersection",
            method="pgd",
            max_iterations=80,
            step_size=0.08,
            seed=0,
        )
    )
    assert response.compatible is True
    assert response.trace[-1].violation <= 0.01 or response.status == "completed_with_constraint_violation"


def test_grid_search_lp_matches_simplex_objective():
    problem = get_problem("lp-basic")
    grid_opt = grid_search_minimize(problem, resolution=80)
    response = solve(
        SolveRequest(problem_id="lp-basic", method="simplex", max_iterations=10, step_size=0.1, seed=0)
    )
    assert response.objective_value is not None
    assert abs(response.objective_value - problem.objective(grid_opt)) < 0.05


def test_every_lp_barrier_path_stays_feasible_and_reaches_simplex_corner():
    for problem in PROBLEMS.values():
        if problem.type != "lp":
            continue
        reference = solve(SolveRequest(problem_id=problem.id, method="interior_point"))
        simplex = solve(SolveRequest(problem_id=problem.id, method="simplex"))
        assert reference.status == "optimal"
        assert max(point.violation for point in reference.trace) <= 1e-6
        assert abs(reference.objective_value - simplex.objective_value) <= 1e-6


def test_diet_example_has_real_nutrient_floors():
    problem = get_problem("lp-diet")
    response = solve(SolveRequest(problem_id=problem.id, method="simplex"))
    x, y = response.variables["x"]
    assert not problem.feasible(np.zeros(2))
    assert 2 * x + y >= 4 - 1e-5
    assert x + 2 * y >= 4 - 1e-5
    assert response.objective_value > 0


def test_reference_points_match_paths_and_are_feasible():
    for problem in PROBLEMS.values():
        response = solve(SolveRequest(problem_id=problem.id, method="interior_point"))
        point = response.variables.get("x", response.variables.get("plot_point"))
        assert point is not None
        assert np.allclose(point, response.trace[-1].x, atol=1e-5)
        assert problem.violation(np.array(point)) < 1e-4
        assert max(iterate.violation for iterate in response.trace) < 1e-4


def test_projection_changes_least_squares_path():
    gd = solve(SolveRequest(problem_id="qp-least-squares", method="gd"))
    pgd = solve(SolveRequest(problem_id="qp-least-squares", method="pgd"))
    assert max(point.violation for point in gd.trace) > 0.01
    assert max(point.violation for point in pgd.trace) <= 1e-6


def test_naive_gd_starts_feasible_then_exits():
    response = solve(SolveRequest(problem_id="qcqp-naive-gd", method="gd"))
    assert response.trace[0].violation == 0
    assert max(point.violation for point in response.trace) > 0.01
