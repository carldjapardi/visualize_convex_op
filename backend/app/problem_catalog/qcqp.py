from __future__ import annotations

import numpy as np

from ..problem_definition import ProblemDefinition
from .helpers import circle_boundary, ellipse_boundary


def _qcqp_constraints_intersection(x: np.ndarray) -> tuple[float, float]:
    circle = x[0] ** 2 + x[1] ** 2 - 4.0
    ellipse = ((x[0] + 0.8) ** 2 / 4.0) + ((x[1] - 0.2) ** 2 / 1.5) - 1.0
    return float(circle), float(ellipse)


def _project_qcqp_intersection(x: np.ndarray) -> np.ndarray:
    point = x.astype(float).copy()
    norm = np.linalg.norm(point)
    if norm > 2.0:
        point *= 2.0 / norm
    shifted = np.array([(point[0] + 0.8) / 2.0, (point[1] - 0.2) / np.sqrt(1.5)])
    scaled_norm = np.linalg.norm(shifted)
    if scaled_norm > 1.0:
        shifted /= scaled_norm
        point = np.array([2.0 * shifted[0] - 0.8, np.sqrt(1.5) * shifted[1] + 0.2])
    norm = np.linalg.norm(point)
    if norm > 2.0:
        point *= 2.0 / norm
    return point


def _project_disk(x: np.ndarray, radius: float) -> np.ndarray:
    point = x.astype(float).copy()
    norm = np.linalg.norm(point)
    if norm > radius:
        point *= radius / norm
    return point


def _project_disk_cap(x: np.ndarray) -> np.ndarray:
    point = _project_disk(x, 1.5)
    if point.sum() < -0.5:
        point = point + ((-0.5 - point.sum()) / 2.0) * np.ones(2)
    return _project_disk(point, 1.5)


def build_qcqp_problems() -> dict[str, ProblemDefinition]:
    center = np.array([1.4, 0.35])

    return {
        "qcqp-intersection": ProblemDefinition(
            id="qcqp-intersection",
            type="qcqp",
            name="QCQP: Quadratic Constraint Intersection",
            description="A convex QCQP with circular and elliptical constraints; compare GD with PGD on this same set.",
            dimension=2,
            variables=["x", "y"],
            compatible_methods=["interior_point", "gd", "sgd", "pgd", "newton"],
            constraints=["x^2 + y^2 <= 4", "(x + 0.8)^2 / 4 + (y - 0.2)^2 / 1.5 <= 1"],
            bounds=((-2.3, 2.3), (-2.3, 2.3)),
            initial_point=np.array([-1.2, 0.1]),
            objective=lambda x: float((x - center) @ (x - center)),
            gradient=lambda x: 2.0 * (x - center),
            hessian=lambda _: 2.0 * np.eye(2),
            feasible=lambda x: all(v <= 1e-9 for v in _qcqp_constraints_intersection(x)),
            violation=lambda x: float(max(0.0, *_qcqp_constraints_intersection(x))),
            project=_project_qcqp_intersection,
            boundaries_fn=lambda: [
                circle_boundary(2.0),
                ellipse_boundary(-0.8, 0.2, 2.0, np.sqrt(1.5)),
            ],
            annotations=["QCQP (quadratic objective and quadratic constraints)."],
        ),
        "qcqp-disk-cap": ProblemDefinition(
            id="qcqp-disk-cap",
            type="qcqp",
            name="QCQP: Disk With Linear Cap",
            description="A convex quadratic objective over a trust-region disk clipped by a linear floor.",
            dimension=2,
            variables=["x", "y"],
            compatible_methods=["interior_point", "gd", "sgd", "pgd", "newton"],
            constraints=["x^2 + y^2 <= 2.25", "x + y >= -0.5"],
            bounds=((-1.7, 1.7), (-1.7, 1.7)),
            initial_point=np.array([0.2, 0.2]),
            objective=lambda x: float((x[0] - 0.6) ** 2 + 0.6 * (x[1] + 0.8) ** 2),
            gradient=lambda x: np.array([2.0 * (x[0] - 0.6), 1.2 * (x[1] + 0.8)]),
            hessian=lambda _: np.array([[2.0, 0.0], [0.0, 1.2]]),
            feasible=lambda x: bool(x[0] ** 2 + x[1] ** 2 <= 2.25 + 1e-9 and x.sum() >= -0.5 - 1e-9),
            violation=lambda x: float(max(0.0, x[0] ** 2 + x[1] ** 2 - 2.25, -0.5 - x.sum())),
            project=_project_disk_cap,
            boundaries_fn=lambda: [
                circle_boundary(1.5),
                {
                    "name": "x + y = -0.5",
                    "x": [(-1 - np.sqrt(17)) / 4, (-1 + np.sqrt(17)) / 4],
                    "y": [(-1 + np.sqrt(17)) / 4, (-1 - np.sqrt(17)) / 4],
                },
            ],
        ),
        "qcqp-trust-region": ProblemDefinition(
            id="qcqp-trust-region",
            type="qcqp",
            name="QCQP: Trust Region",
            description="Linear objective over a spherical quadratic constraint (QCQP, not LP).",
            dimension=2,
            variables=["x", "y"],
            compatible_methods=["interior_point", "pgd"],
            constraints=["x^2 + y^2 <= 1"],
            bounds=((-1.3, 1.3), (-1.3, 1.3)),
            initial_point=np.array([0.7, 0.6]),
            objective=lambda x: float(-2.0 * x[0] - x[1]),
            gradient=lambda _: np.array([-2.0, -1.0]),
            hessian=lambda _: np.zeros((2, 2)),
            feasible=lambda x: bool(x[0] ** 2 + x[1] ** 2 <= 1.0 + 1e-9),
            violation=lambda x: float(max(0.0, x[0] ** 2 + x[1] ** 2 - 1.0)),
            project=lambda x: _project_disk(x, 1.0),
            boundaries_fn=lambda: [circle_boundary(1.0)],
            trace_message="Linear objective on a quadratic trust region — PGD projects onto the disk.",
        ),
        "qcqp-ellipse-box": ProblemDefinition(
            id="qcqp-ellipse-box",
            type="qcqp",
            name="QCQP: Ellipse and Box",
            description="Minimize a quadratic objective over the intersection of a convex ellipse and a box.",
            dimension=2,
            variables=["x", "y"],
            compatible_methods=["interior_point", "gd", "sgd", "pgd", "newton"],
            constraints=["(x - 0.1)^2 / 2.25 + (y - 0.2)^2 / 1.44 <= 1", "-1 <= x <= 2", "-1 <= y <= 2"],
            bounds=((-1.5, 2.5), (-1.5, 2.5)),
            initial_point=np.array([0.8, 0.4]),
            objective=lambda x: float((x[0] - 1.8) ** 2 + (x[1] - 1.6) ** 2),
            gradient=lambda x: np.array([2.0 * (x[0] - 1.8), 2.0 * (x[1] - 1.6)]),
            hessian=lambda _: 2.0 * np.eye(2),
            feasible=lambda x: bool(
                (x[0] - 0.1) ** 2 / 2.25 + (x[1] - 0.2) ** 2 / 1.44 <= 1.0 + 1e-9
                and -1.0 <= x[0] <= 2.0 and -1.0 <= x[1] <= 2.0
            ),
            violation=lambda x: float(max(0.0, (x[0] - 0.1) ** 2 / 2.25 + (x[1] - 0.2) ** 2 / 1.44 - 1.0, -1.0 - x[0], x[0] - 2.0, -1.0 - x[1], x[1] - 2.0)),
            project=_project_ellipse_box,
            boundaries_fn=lambda: [ellipse_boundary(0.1, 0.2, 1.5, 1.2)],
            default_step_size=0.08,
        ),
    }


def _project_ellipse_box(x: np.ndarray) -> np.ndarray:
    point = np.clip(x, [-1.0, -1.0], [2.0, 2.0]).astype(float)
    center = np.array([0.1, 0.2])
    scaled = (point - center) / np.array([1.5, 1.2])
    norm = np.linalg.norm(scaled)
    if norm > 1.0:
        point = center + (point - center) / norm
    return point
