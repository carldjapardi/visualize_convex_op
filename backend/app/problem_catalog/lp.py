from __future__ import annotations

import numpy as np

from ..problem_definition import LPData, ProblemDefinition


def _lp_boundaries_basic() -> list[dict[str, object]]:
    return [
        {"name": "x + y = 4", "x": [0, 4], "y": [4, 0]},
        {"name": "x = 3", "x": [3, 3], "y": [0, 2]},
        {"name": "y = 2", "x": [0, 3], "y": [2, 2]},
    ]


def _lp_boundaries_ratio() -> list[dict[str, object]]:
    return [
        {"name": "x + y = 5", "x": [0, 5], "y": [5, 0]},
        {"name": "y = 0.5x", "x": [0, 5], "y": [0, 2.5]},
        {"name": "y = 3", "x": [0, 5], "y": [3, 3]},
    ]


def _lp_boundaries_diet() -> list[dict[str, object]]:
    return [
        {"name": "2x + y = 4", "x": [0, 2], "y": [4, 0]},
        {"name": "x + 2y = 4", "x": [0, 4], "y": [2, 0]},
        {"name": "x + y = 5", "x": [0, 5], "y": [5, 0]},
    ]


def _lp_boundaries_delivery() -> list[dict[str, object]]:
    return [
        {"name": "x + y = 4", "x": [0, 4], "y": [4, 0]},
        {"name": "x + y = 5", "x": [0, 5], "y": [5, 0]},
        {"name": "x = 3", "x": [3, 3], "y": [0, 5]},
        {"name": "y = 3.5", "x": [0, 5], "y": [3.5, 3.5]},
    ]


def _lp_boundaries_alternate() -> list[dict[str, object]]:
    return [
        {"name": "x + y = 4", "x": [0, 4], "y": [4, 0]},
        {"name": "x = 3.2", "x": [3.2, 3.2], "y": [0, 4]},
        {"name": "y = 2.7", "x": [0, 4], "y": [2.7, 2.7]},
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
        "lp-ratio-blend": ProblemDefinition(
            id="lp-ratio-blend",
            type="lp",
            name="LP: Ratio-Constrained Blend",
            description="Choose two ingredients while keeping at least one unit of y for every two units of x.",
            dimension=2,
            variables=["x", "y"],
            compatible_methods=["simplex", "interior_point"],
            constraints=["x >= 0", "y >= 0", "x + y <= 5", "y >= 0.5x", "y <= 3"],
            bounds=((-0.4, 5.4), (-0.4, 3.5)),
            initial_point=np.array([1.0, 1.0]),
            objective=lambda x: float(-4.0 * x[0] - 2.0 * x[1]),
            gradient=lambda _: np.array([-4.0, -2.0]),
            feasible=lambda x: bool(
                x[0] >= -1e-9
                and x[1] >= -1e-9
                and x.sum() <= 5.0 + 1e-9
                and x[1] >= 0.5 * x[0] - 1e-9
                and x[1] <= 3.0 + 1e-9
            ),
            violation=lambda x: float(
                max(0.0, -x[0], -x[1], x.sum() - 5.0, 0.5 * x[0] - x[1], x[1] - 3.0)
            ),
            boundaries_fn=_lp_boundaries_ratio,
            lp_data=LPData(
                c=np.array([-4.0, -2.0]),
                a_ub=np.array([[1.0, 1.0], [0.5, -1.0]]),
                b_ub=np.array([5.0, 0.0]),
                bounds=[(0.0, None), (0.0, 3.0)],
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
        "lp-delivery-window": ProblemDefinition(
            id="lp-delivery-window",
            type="lp",
            name="LP: Delivery Window",
            description="Meet a minimum shipment without exceeding a maximum, then choose the cheaper route mix.",
            dimension=2,
            variables=["route 1", "route 2"],
            compatible_methods=["simplex", "interior_point"],
            constraints=["x >= 0", "y >= 0", "4 <= x + y <= 5", "x <= 3", "y <= 3.5"],
            bounds=((-0.4, 5.4), (-0.4, 5.4)),
            initial_point=np.array([2.0, 2.5]),
            objective=lambda x: float(2.0 * x[0] + 4.0 * x[1]),
            gradient=lambda _: np.array([2.0, 4.0]),
            feasible=lambda x: bool(
                x[0] >= -1e-9 and x[1] >= -1e-9 and 4 - 1e-9 <= x.sum() <= 5 + 1e-9
                and x[0] <= 3 + 1e-9 and x[1] <= 3.5 + 1e-9
            ),
            violation=lambda x: float(max(0.0, -x[0], -x[1], 4 - x.sum(), x.sum() - 5, x[0] - 3, x[1] - 3.5)),
            boundaries_fn=_lp_boundaries_delivery,
            lp_data=LPData(
                c=np.array([2.0, 4.0]),
                a_ub=np.array([[-1.0, -1.0], [1.0, 1.0]]),
                b_ub=np.array([-4.0, 5.0]),
                bounds=[(0.0, 3.0), (0.0, 3.5)],
            ),
        ),
        "lp-alternate-optima": ProblemDefinition(
            id="lp-alternate-optima",
            type="lp",
            name="LP: Alternate Optima",
            description="The objective lines up with one feasible edge, so many allocations tie for best.",
            dimension=2,
            variables=["allocation x", "allocation y"],
            compatible_methods=["simplex", "interior_point"],
            constraints=["x >= 0", "y >= 0", "x + y <= 4", "x <= 3.2", "y <= 2.7"],
            bounds=((-0.4, 4.4), (-0.4, 4.0)),
            initial_point=np.array([1.0, 1.0]),
            annotations=["Every point on the segment from (1.3, 2.7) to (3.2, 0.8) has objective −4."],
            objective=lambda x: float(-x[0] - x[1]),
            gradient=lambda _: np.array([-1.0, -1.0]),
            feasible=lambda x: bool(
                x[0] >= -1e-9 and x[1] >= -1e-9 and x.sum() <= 4 + 1e-9
                and x[0] <= 3.2 + 1e-9 and x[1] <= 2.7 + 1e-9
            ),
            violation=lambda x: float(max(0.0, -x[0], -x[1], x.sum() - 4, x[0] - 3.2, x[1] - 2.7)),
            boundaries_fn=_lp_boundaries_alternate,
            lp_data=LPData(
                c=np.array([-1.0, -1.0]),
                a_ub=np.array([[1.0, 1.0]]),
                b_ub=np.array([4.0]),
                bounds=[(0.0, 3.2), (0.0, 2.7)],
            ),
        ),
    }
