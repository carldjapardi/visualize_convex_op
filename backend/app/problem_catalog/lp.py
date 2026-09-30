from __future__ import annotations

import numpy as np

from ..problem_definition import LPData, ProblemDefinition


def _lp_boundaries_basic() -> list[dict[str, object]]:
    return [
        {"name": "x + y = 4", "x": [0, 4], "y": [4, 0]},
        {"name": "x = 3", "x": [3, 3], "y": [0, 2]},
        {"name": "y = 2", "x": [0, 3], "y": [2, 2]},
    ]


def _lp_boundaries_warehouse() -> list[dict[str, object]]:
    return [
        {"name": "2x + y = 8", "x": [2, 4], "y": [4, 0]},
        {"name": "x + 3y = 9", "x": [0, 4.2], "y": [3, 1.6]},
        {"name": "x = 3.5", "x": [3.5, 3.5], "y": [0, 1]},
    ]


def _lp_boundaries_diet() -> list[dict[str, object]]:
    return [
        {"name": "2x + y = 4", "x": [0, 2], "y": [4, 0]},
        {"name": "x + 2y = 4", "x": [0, 4], "y": [2, 0]},
        {"name": "x + y = 5", "x": [0, 5], "y": [5, 0]},
    ]


def _lp_boundaries_transport() -> list[dict[str, object]]:
    return [
        {"name": "x + y = 5", "x": [0, 5], "y": [5, 0]},
        {"name": "2x + y = 7", "x": [0, 3.5], "y": [7, 0]},
    ]


def _lp_boundaries_flow() -> list[dict[str, object]]:
    return [
        {"name": "x + 2y = 6", "x": [0, 6], "y": [3, 0]},
        {"name": "3x + y = 6", "x": [0, 2], "y": [6, 0]},
    ]


def build_lp_problems() -> dict[str, ProblemDefinition]:
    return {
        "lp-basic": ProblemDefinition(
            id="lp-basic",
            type="lp",
            name="LP: Production Mix",
            description="A two-variable linear program with a polygonal feasible region.",
            dimension=2,
            variables=["x", "y"],
            compatible_methods=["simplex", "interior_point"],
            constraints=["x >= 0", "y >= 0", "x + y <= 4", "x <= 3", "y <= 2"],
            bounds=((-0.4, 4.2), (-0.4, 3.0)),
            initial_point=np.array([0.5, 0.5]),
            objective=lambda x: float(-3.0 * x[0] - 2.0 * x[1]),
            gradient=lambda _: np.array([-3.0, -2.0]),
            feasible=lambda x: bool(
                x[0] >= -1e-9 and x[1] >= -1e-9 and x.sum() <= 4.0 + 1e-9 and x[0] <= 3.0 + 1e-9 and x[1] <= 2.0 + 1e-9
            ),
            violation=lambda x: float(max(0.0, -x[0], -x[1], x.sum() - 4.0, x[0] - 3.0, x[1] - 2.0)),
            project=lambda x: np.array([min(max(x[0], 0.0), 3.0), min(max(x[1], 0.0), 2.0)]),
            boundaries_fn=_lp_boundaries_basic,
            lp_data=LPData(
                c=np.array([-3.0, -2.0]),
                a_ub=np.array([[1.0, 1.0]]),
                b_ub=np.array([4.0]),
                bounds=[(0.0, 3.0), (0.0, 2.0)],
            ),
        ),
        "lp-warehouse": ProblemDefinition(
            id="lp-warehouse",
            type="lp",
            name="LP: Warehouse Allocation",
            description="A linear program with two shared resource budgets and a different optimal corner.",
            dimension=2,
            variables=["x", "y"],
            compatible_methods=["simplex", "interior_point"],
            constraints=["x >= 0", "y >= 0", "2x + y <= 8", "x + 3y <= 9", "x <= 3.5"],
            bounds=((-0.4, 4.4), (-0.4, 4.2)),
            initial_point=np.array([0.6, 0.6]),
            objective=lambda x: float(-4.0 * x[0] - 3.0 * x[1]),
            gradient=lambda _: np.array([-4.0, -3.0]),
            feasible=lambda x: bool(
                x[0] >= -1e-9
                and x[1] >= -1e-9
                and 2.0 * x[0] + x[1] <= 8.0 + 1e-9
                and x[0] + 3.0 * x[1] <= 9.0 + 1e-9
                and x[0] <= 3.5 + 1e-9
            ),
            violation=lambda x: float(
                max(0.0, -x[0], -x[1], 2.0 * x[0] + x[1] - 8.0, x[0] + 3.0 * x[1] - 9.0, x[0] - 3.5)
            ),
            project=lambda x: np.array([min(max(x[0], 0.0), 3.5), max(x[1], 0.0)]),
            boundaries_fn=_lp_boundaries_warehouse,
            lp_data=LPData(
                c=np.array([-4.0, -3.0]),
                a_ub=np.array([[2.0, 1.0], [1.0, 3.0]]),
                b_ub=np.array([8.0, 9.0]),
                bounds=[(0.0, 3.5), (0.0, None)],
            ),
        ),
        "lp-diet": ProblemDefinition(
            id="lp-diet",
            type="lp",
            name="LP: Diet Planning",
            description="Find the least-cost mix that meets two nutrient minimums without exceeding a total intake cap.",
            dimension=2,
            variables=["meals A", "meals B"],
            compatible_methods=["simplex", "interior_point"],
            constraints=["x >= 0", "y >= 0", "2x + y >= 4", "x + 2y >= 4", "x + y <= 5"],
            bounds=((-0.4, 5.5), (-0.4, 5.5)),
            initial_point=np.array([2.0, 2.0]),
            objective=lambda x: float(2.0 * x[0] + 1.5 * x[1]),
            gradient=lambda _: np.array([2.0, 1.5]),
            feasible=lambda x: bool(
                x[0] >= -1e-9
                and x[1] >= -1e-9
                and 2 * x[0] + x[1] >= 4 - 1e-9
                and x[0] + 2 * x[1] >= 4 - 1e-9
                and x.sum() <= 5 + 1e-9
            ),
            violation=lambda x: float(
                max(0.0, -x[0], -x[1], 4 - 2 * x[0] - x[1], 4 - x[0] - 2 * x[1], x.sum() - 5)
            ),
            boundaries_fn=_lp_boundaries_diet,
            lp_data=LPData(
                c=np.array([2.0, 1.5]),
                a_ub=np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 1.0]]),
                b_ub=np.array([-4.0, -4.0, 5.0]),
                bounds=[(0.0, None), (0.0, None)],
            ),
        ),
        "lp-transport": ProblemDefinition(
            id="lp-transport",
            type="lp",
            name="LP: Transport Balance",
            description="Ship goods along two routes with balance caps and a total shipment limit.",
            dimension=2,
            variables=["route 1", "route 2"],
            compatible_methods=["simplex", "interior_point"],
            constraints=["x >= 0", "y >= 0", "x + y <= 5", "2x + y <= 7"],
            bounds=((-0.4, 5.5), (-0.4, 5.5)),
            initial_point=np.array([0.8, 0.8]),
            objective=lambda x: float(-5.0 * x[0] - 4.0 * x[1]),
            gradient=lambda _: np.array([-5.0, -4.0]),
            feasible=lambda x: bool(
                x[0] >= -1e-9 and x[1] >= -1e-9 and x.sum() <= 5 + 1e-9 and 2 * x[0] + x[1] <= 7 + 1e-9
            ),
            violation=lambda x: float(max(0.0, -x[0], -x[1], x.sum() - 5, 2 * x[0] + x[1] - 7)),
            boundaries_fn=_lp_boundaries_transport,
            lp_data=LPData(
                c=np.array([-5.0, -4.0]),
                a_ub=np.array([[1.0, 1.0], [2.0, 1.0]]),
                b_ub=np.array([5.0, 7.0]),
                bounds=[(0.0, None), (0.0, None)],
            ),
        ),
        "lp-max-flow-slice": ProblemDefinition(
            id="lp-max-flow-slice",
            type="lp",
            name="LP: Max Flow Slice",
            description="A 2D capacity slice of a flow problem with intersecting linear caps.",
            dimension=2,
            variables=["flow x", "flow y"],
            compatible_methods=["simplex", "interior_point"],
            constraints=["x >= 0", "y >= 0", "x + 2y <= 6", "3x + y <= 6"],
            bounds=((-0.4, 3.5), (-0.4, 3.5)),
            initial_point=np.array([0.4, 0.4]),
            objective=lambda x: float(-x[0] - x[1]),
            gradient=lambda _: np.array([-1.0, -1.0]),
            feasible=lambda x: bool(
                x[0] >= -1e-9 and x[1] >= -1e-9 and x[0] + 2 * x[1] <= 6 + 1e-9 and 3 * x[0] + x[1] <= 6 + 1e-9
            ),
            violation=lambda x: float(max(0.0, -x[0], -x[1], x[0] + 2 * x[1] - 6, 3 * x[0] + x[1] - 6)),
            boundaries_fn=_lp_boundaries_flow,
            lp_data=LPData(
                c=np.array([-1.0, -1.0]),
                a_ub=np.array([[1.0, 2.0], [3.0, 1.0]]),
                b_ub=np.array([6.0, 6.0]),
                bounds=[(0.0, None), (0.0, None)],
            ),
        ),
    }
