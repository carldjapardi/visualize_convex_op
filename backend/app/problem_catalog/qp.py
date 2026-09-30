from __future__ import annotations

import numpy as np

from ..problem_definition import ProblemDefinition
from .helpers import project_halfspace_sum, quad_form, quad_gradient


def build_qp_problems() -> dict[str, ProblemDefinition]:
    q_con = np.array([[3.0, 0.6], [0.6, 1.4]])
    v_con = np.array([-2.0, -1.0])
    q_port = np.array([[2.4, 0.8], [0.8, 1.7]])
    v_port = np.array([-1.1, -1.5])
    q_ls = np.array([[4.0, 0.0], [0.0, 2.5]])
    v_ls = np.array([3.0, -0.5])
    q_elas = np.array([[5.0, 0.0], [0.0, 1.2]])
    v_elas = np.array([-1.2, -0.8])
    q_strong = np.array([[2.0, 0.3], [0.3, 3.0]])
    v_strong = np.array([-1.0, -1.2])

    lower = np.array([-1.0, -1.0])
    upper = np.array([3.0, 3.0])

    return {
        "qp-constrained": ProblemDefinition(
            id="qp-constrained",
            type="qp",
            name="QP: Constrained Quadratic Bowl",
            description="A convex quadratic objective over a simple polytope.",
            dimension=2,
            variables=["x", "y"],
            compatible_methods=["interior_point", "gd", "sgd", "pgd", "newton"],
            constraints=["-1 <= x <= 3", "-1 <= y <= 3", "x + y <= 3.5"],
            bounds=((-1.4, 3.4), (-1.4, 3.4)),
            initial_point=np.array([2.5, 0.8]),
            objective=lambda x: quad_form(q_con, v_con, x),
            gradient=lambda x: quad_gradient(q_con, v_con, x),
            hessian=lambda _: q_con.copy(),
            feasible=lambda x: bool(-1.0 <= x[0] <= 3.0 and -1.0 <= x[1] <= 3.0 and x.sum() <= 3.5 + 1e-9),
            violation=lambda x: float(max(0.0, -1.0 - x[0], x[0] - 3.0, -1.0 - x[1], x[1] - 3.0, x.sum() - 3.5)),
            project=lambda x: project_halfspace_sum(x, lower, upper, None, 3.5),
            boundaries_fn=lambda: [{"name": "x + y = 3.5", "x": [-1, 3], "y": [4.5, 0.5]}],
        ),
        "qp-portfolio": ProblemDefinition(
            id="qp-portfolio",
            type="qp",
            name="QP: Portfolio Variance Tradeoff",
            description="A convex quadratic risk model over bounded allocation weights.",
            dimension=2,
            variables=["asset A", "asset B"],
            compatible_methods=["interior_point", "gd", "sgd", "pgd", "newton"],
            constraints=["0 <= x <= 1.4", "0 <= y <= 1.4", "0.35 <= x + y <= 1.6"],
            bounds=((-0.2, 1.8), (-0.2, 1.8)),
            initial_point=np.array([1.25, 0.3]),
            objective=lambda x: quad_form(q_port, v_port, x),
            gradient=lambda x: quad_gradient(q_port, v_port, x),
            hessian=lambda _: q_port.copy(),
            feasible=lambda x: bool(0.0 <= x[0] <= 1.4 and 0.0 <= x[1] <= 1.4 and 0.35 - 1e-9 <= x.sum() <= 1.6 + 1e-9),
            violation=lambda x: float(max(0.0, -x[0], x[0] - 1.4, -x[1], x[1] - 1.4, 0.35 - x.sum(), x.sum() - 1.6)),
            project=lambda x: project_halfspace_sum(
                x, np.array([0.0, 0.0]), np.array([1.4, 1.4]), 0.35, 1.6
            ),
            boundaries_fn=lambda: [
                {"name": "x + y = 0.35", "x": [0, 0.35], "y": [0.35, 0]},
                {"name": "x + y = 1.6", "x": [0.2, 1.4], "y": [1.4, 0.2]},
            ],
        ),
        "qp-least-squares": ProblemDefinition(
            id="qp-least-squares",
            type="qp",
            name="QP: Least Squares Box",
            description="Convex least-squares objective with only box constraints — compare GD and PGD.",
            dimension=2,
            variables=["x", "y"],
            compatible_methods=["interior_point", "gd", "sgd", "pgd", "newton"],
            constraints=["-0.5 <= x <= 2", "-0.5 <= y <= 2"],
            bounds=((-1.0, 2.5), (-1.0, 2.5)),
            initial_point=np.array([1.8, 1.6]),
            objective=lambda x: quad_form(q_ls, v_ls, x),
            gradient=lambda x: quad_gradient(q_ls, v_ls, x),
            hessian=lambda _: q_ls.copy(),
            feasible=lambda x: bool(-0.5 <= x[0] <= 2.0 and -0.5 <= x[1] <= 2.0),
            violation=lambda x: float(max(0.0, -0.5 - x[0], x[0] - 2.0, -0.5 - x[1], x[1] - 2.0)),
            project=lambda x: np.clip(x, [-0.5, -0.5], [2.0, 2.0]),
            default_step_size=0.08,
        ),
        "qp-elastic-net-slice": ProblemDefinition(
            id="qp-elastic-net-slice",
            type="qp",
            name="QP: Anisotropic Quadratic",
            description="A diagonal quadratic with different curvature along the two axes.",
            dimension=2,
            variables=["x", "y"],
            compatible_methods=["interior_point", "gd", "sgd", "pgd", "newton"],
            constraints=["0 <= x <= 1.5", "0 <= y <= 1.5"],
            bounds=((-0.2, 1.8), (-0.2, 1.8)),
            initial_point=np.array([1.2, 1.2]),
            objective=lambda x: quad_form(q_elas, v_elas, x),
            gradient=lambda x: quad_gradient(q_elas, v_elas, x),
            hessian=lambda _: q_elas.copy(),
            feasible=lambda x: bool(0.0 <= x[0] <= 1.5 and 0.0 <= x[1] <= 1.5),
            violation=lambda x: float(max(0.0, -x[0], x[0] - 1.5, -x[1], x[1] - 1.5)),
            project=lambda x: np.clip(x, [0.0, 0.0], [1.5, 1.5]),
            default_step_size=0.06,
        ),
        "qp-saddle-slice": ProblemDefinition(
            id="qp-saddle-slice",
            type="qp",
            name="QP: Strongly Convex Slice",
            description="Well-conditioned convex quadratic — Newton converges quickly inside the box.",
            dimension=2,
            variables=["x", "y"],
            compatible_methods=["interior_point", "gd", "sgd", "pgd", "newton"],
            constraints=["-1 <= x <= 2", "-1 <= y <= 2"],
            bounds=((-1.4, 2.4), (-1.4, 2.4)),
            initial_point=np.array([1.5, -0.5]),
            objective=lambda x: quad_form(q_strong, v_strong, x),
            gradient=lambda x: quad_gradient(q_strong, v_strong, x),
            hessian=lambda _: q_strong.copy(),
            feasible=lambda x: bool(-1.0 <= x[0] <= 2.0 and -1.0 <= x[1] <= 2.0),
            violation=lambda x: float(max(0.0, -1.0 - x[0], x[0] - 2.0, -1.0 - x[1], x[1] - 2.0)),
            project=lambda x: np.clip(x, [-1.0, -1.0], [2.0, 2.0]),
            default_step_size=0.1,
        ),
    }
