from __future__ import annotations

import pytest

from app.compatibility import allowed_methods, validate_compatible_methods
from app.problems import PROBLEMS, get_problem
from app.methods import solve
from app.schemas import SolveRequest


def test_twenty_five_problems():
    assert len(PROBLEMS) == 25
    by_type: dict[str, int] = {}
    for problem in PROBLEMS.values():
        by_type[problem.type] = by_type.get(problem.type, 0) + 1
    assert by_type == {"lp": 5, "qp": 5, "qcqp": 5, "socp": 5, "sdp": 5}


def test_simplex_only_on_lp():
    for problem in PROBLEMS.values():
        if "simplex" in problem.compatible_methods:
            assert problem.type == "lp"
        if problem.type != "lp":
            assert "simplex" not in problem.compatible_methods


def test_type_allowed_methods_subset():
    for problem in PROBLEMS.values():
        allowed = allowed_methods(problem.type)
        assert set(problem.compatible_methods).issubset(allowed)


def test_invalid_method_registration():
    with pytest.raises(ValueError):
        validate_compatible_methods("lp", ["gd"])


def test_unsupported_solve_pair():
    response = solve(
        SolveRequest(problem_id="qp-constrained", method="simplex", max_iterations=10, step_size=0.1, seed=0)
    )
    assert response.status == "unsupported"
    assert response.compatible is False


def test_lp_simplex_solves():
    response = solve(
        SolveRequest(problem_id="lp-basic", method="simplex", max_iterations=10, step_size=0.1, seed=0)
    )
    assert response.compatible is True
    assert response.status == "optimal"
    assert response.solver == "Educational 2D simplex"
    assert len(response.trace) >= 1


def test_qcqp_no_simplex_in_catalog():
    for pid in ("qcqp-intersection", "qcqp-naive-gd"):
        problem = get_problem(pid)
        assert "simplex" not in problem.compatible_methods
