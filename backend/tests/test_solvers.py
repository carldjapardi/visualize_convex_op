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


def test_distinct_lp_geometries_and_alternate_optima():
    ratio = get_problem("lp-ratio-blend")
    ratio_result = solve(SolveRequest(problem_id=ratio.id, method="simplex"))
    assert np.allclose(ratio_result.variables["x"], [10 / 3, 5 / 3], atol=1e-5)
    assert ratio.feasible(np.array(ratio_result.variables["x"]))

    window = get_problem("lp-delivery-window")
    window_result = solve(SolveRequest(problem_id=window.id, method="simplex"))
    assert not window.feasible(np.zeros(2))
    assert np.allclose(window_result.variables["x"], [3, 1], atol=1e-5)

    alternate = get_problem("lp-alternate-optima")
    left = np.array([1.3, 2.7])
    right = np.array([3.2, 0.8])
    assert alternate.feasible(left) and alternate.feasible(right)
    assert alternate.objective(left) == alternate.objective(right) == -4
    result = solve(SolveRequest(problem_id=alternate.id, method="simplex"))
    assert abs(result.objective_value + 4) < 1e-8


def test_reference_points_match_paths_and_are_feasible():
    for problem in PROBLEMS.values():
        if problem.type == "nonconvex":
            continue
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


def test_qcqp_gd_starts_feasible_then_exits():
    response = solve(SolveRequest(problem_id="qcqp-intersection", method="gd"))
    assert response.trace[0].violation == 0
    assert max(point.violation for point in response.trace) > 0.01


def test_nonconvex_start_changes_destination():
    problem = get_problem("nonconvex-double-well")
    right = solve(SolveRequest(problem_id=problem.id, method="gd", step_size=problem.default_step_size,
                               max_iterations=problem.default_max_iterations, start_point=(0.3, 1.15)))
    left = solve(SolveRequest(problem_id=problem.id, method="gd", step_size=problem.default_step_size,
                              max_iterations=problem.default_max_iterations, start_point=(-0.3, 1.15)))
    assert right.trace[-1].x[0] > 0.9
    assert left.trace[-1].x[0] < -0.9
    assert left.objective_value < right.objective_value - 0.4


def test_nonconvex_derivatives_and_default_paths_are_finite():
    for problem in PROBLEMS.values():
        if problem.type != "nonconvex":
            continue
        x = problem.initial_point.astype(float)
        epsilon = 1e-5
        numerical = np.array([
            (problem.objective(x + epsilon * np.eye(2)[i]) - problem.objective(x - epsilon * np.eye(2)[i])) / (2 * epsilon)
            for i in range(2)
        ])
        assert np.allclose(problem.gradient(x), numerical, atol=1e-4)
        for method in problem.compatible_methods:
            response = solve(SolveRequest(problem_id=problem.id, method=method,
                                          step_size=problem.default_step_size,
                                          max_iterations=problem.default_max_iterations))
            assert response.compatible and response.trace
            assert all(np.isfinite(point.objective) and np.all(np.isfinite(point.x)) for point in response.trace)


def test_nonconvex_large_step_stops_cleanly_at_plot_window():
    response = solve(SolveRequest(problem_id="nonconvex-himmelblau", method="gd", step_size=2.0))
    assert response.status == "left_plot_window"
    assert len(response.trace) == 1
    assert np.isfinite(response.objective_value)
