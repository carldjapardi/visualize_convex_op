from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cvxpy as cp
import numpy as np

from .schemas import GeometryResponse, MethodId, ProblemSummary, ProblemType


ArrayFn = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class ProblemDefinition:
    id: str
    type: ProblemType
    name: str
    description: str
    dimension: int
    variables: list[str]
    compatible_methods: list[MethodId]
    constraints: list[str]
    bounds: tuple[tuple[float, float], tuple[float, float]]
    initial_point: np.ndarray
    objective: ArrayFn
    gradient: Callable[[np.ndarray], np.ndarray] | None
    hessian: Callable[[np.ndarray], np.ndarray] | None
    feasible: Callable[[np.ndarray], bool]
    violation: Callable[[np.ndarray], float]
    project: Callable[[np.ndarray], np.ndarray] | None = None

    def summary(self) -> ProblemSummary:
        return ProblemSummary(
            id=self.id,
            type=self.type,
            name=self.name,
            description=self.description,
            dimension=self.dimension,
            variables=self.variables,
            compatible_methods=self.compatible_methods,
            constraints=self.constraints,
        )


METHODS: dict[MethodId, dict[str, object]] = {
    "simplex": {
        "name": "Simplex",
        "description": "LP vertex-to-vertex optimization through SciPy HiGHS dual simplex.",
        "best_for": ["lp"],
    },
    "interior_point": {
        "name": "Interior Point / Conic Solver",
        "description": "Primal-dual style conic optimization through CVXPY-backed solvers.",
        "best_for": ["lp", "qp", "qcqp", "socp", "sdp"],
    },
    "gd": {
        "name": "Gradient Descent",
        "description": "First-order descent trace for smooth objective landscapes.",
        "best_for": ["qp", "qcqp"],
    },
    "sgd": {
        "name": "Stochastic Gradient Descent",
        "description": "Noisy first-order descent trace for smooth objective landscapes.",
        "best_for": ["qp", "qcqp"],
    },
    "pgd": {
        "name": "Projected Gradient Descent",
        "description": "Gradient descent followed by projection onto the demo feasible set.",
        "best_for": ["qp", "qcqp"],
    },
    "newton": {
        "name": "Newton's Method",
        "description": "Second-order descent using the local Hessian when it is available.",
        "best_for": ["qp", "qcqp"],
    },
}


def method_summary(method_id: MethodId):
    from .schemas import MethodSummary

    item = METHODS[method_id]
    return MethodSummary(
        id=method_id,
        name=str(item["name"]),
        description=str(item["description"]),
        best_for=item["best_for"],  # type: ignore[arg-type]
    )


def _qp_objective(x: np.ndarray) -> float:
    q_mat = np.array([[3.0, 0.6], [0.6, 1.4]])
    q_vec = np.array([-2.0, -1.0])
    return float(0.5 * x @ q_mat @ x + q_vec @ x)


def _qp_gradient(x: np.ndarray) -> np.ndarray:
    return np.array([[3.0, 0.6], [0.6, 1.4]]) @ x + np.array([-2.0, -1.0])


def _qp_hessian(_: np.ndarray) -> np.ndarray:
    return np.array([[3.0, 0.6], [0.6, 1.4]])


def _project_qp(x: np.ndarray) -> np.ndarray:
    point = np.clip(x, [-1.0, -1.0], [3.0, 3.0])
    if point.sum() > 3.5:
        point = point - ((point.sum() - 3.5) / 2.0) * np.ones(2)
    return np.clip(point, [-1.0, -1.0], [3.0, 3.0])


def _portfolio_objective(x: np.ndarray) -> float:
    q_mat = np.array([[2.4, 0.8], [0.8, 1.7]])
    q_vec = np.array([-1.1, -1.5])
    return float(0.5 * x @ q_mat @ x + q_vec @ x)


def _portfolio_gradient(x: np.ndarray) -> np.ndarray:
    return np.array([[2.4, 0.8], [0.8, 1.7]]) @ x + np.array([-1.1, -1.5])


def _portfolio_hessian(_: np.ndarray) -> np.ndarray:
    return np.array([[2.4, 0.8], [0.8, 1.7]])


def _project_portfolio(x: np.ndarray) -> np.ndarray:
    point = np.clip(x, [0.0, 0.0], [1.4, 1.4])
    if point.sum() > 1.6:
        point = point - ((point.sum() - 1.6) / 2.0) * np.ones(2)
    if point.sum() < 0.35:
        point = point + ((0.35 - point.sum()) / 2.0) * np.ones(2)
    return np.clip(point, [0.0, 0.0], [1.4, 1.4])


def _qcqp_objective(x: np.ndarray) -> float:
    center = np.array([1.4, 0.35])
    diff = x - center
    return float(diff @ diff)


def _qcqp_gradient(x: np.ndarray) -> np.ndarray:
    return 2.0 * (x - np.array([1.4, 0.35]))


def _qcqp_hessian(_: np.ndarray) -> np.ndarray:
    return 2.0 * np.eye(2)


def _qcqp_constraints(x: np.ndarray) -> tuple[float, float]:
    circle = x[0] ** 2 + x[1] ** 2 - 4.0
    ellipse = ((x[0] + 0.8) ** 2 / 4.0) + ((x[1] - 0.2) ** 2 / 1.5) - 1.0
    return float(circle), float(ellipse)


def _project_qcqp(x: np.ndarray) -> np.ndarray:
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


def _qcqp_cap_objective(x: np.ndarray) -> float:
    return float((x[0] - 0.6) ** 2 + 0.6 * (x[1] + 0.8) ** 2)


def _qcqp_cap_gradient(x: np.ndarray) -> np.ndarray:
    return np.array([2.0 * (x[0] - 0.6), 1.2 * (x[1] + 0.8)])


def _qcqp_cap_hessian(_: np.ndarray) -> np.ndarray:
    return np.array([[2.0, 0.0], [0.0, 1.2]])


def _project_qcqp_cap(x: np.ndarray) -> np.ndarray:
    point = x.astype(float).copy()
    if point.sum() < -0.5:
        point = point + ((-0.5 - point.sum()) / 2.0) * np.ones(2)

    norm = np.linalg.norm(point)
    if norm > 1.5:
        point *= 1.5 / norm

    if point.sum() < -0.5:
        point = point + ((-0.5 - point.sum()) / 2.0) * np.ones(2)
    return point


def _socp_rhs(x: np.ndarray) -> float:
    return float(1.85 - 0.25 * x[0] + 0.1 * x[1])


def _socp_objective(x: np.ndarray) -> float:
    return float(-x[0] - 0.7 * x[1])


def _sdp_objective(x: np.ndarray) -> float:
    # x = [a, b] represents X = [[a, b], [b, 1 - a]].
    return float(-x[1] + 0.05 * x[0])


def _sdp_feasible(x: np.ndarray) -> bool:
    a, b = x
    return 0.0 <= a <= 1.0 and b * b <= a * (1.0 - a) + 1e-9


def _sdp_violation(x: np.ndarray) -> float:
    a, b = x
    return float(max(0.0, -a, a - 1.0, b * b - a * (1.0 - a)))


PROBLEMS: dict[str, ProblemDefinition] = {
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
        hessian=lambda _: np.zeros((2, 2)),
        feasible=lambda x: bool(x[0] >= -1e-9 and x[1] >= -1e-9 and x.sum() <= 4.0 + 1e-9 and x[0] <= 3.0 + 1e-9 and x[1] <= 2.0 + 1e-9),
        violation=lambda x: float(max(0.0, -x[0], -x[1], x.sum() - 4.0, x[0] - 3.0, x[1] - 2.0)),
        project=lambda x: np.array([min(max(x[0], 0.0), 3.0), min(max(x[1], 0.0), 2.0)]),
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
        hessian=lambda _: np.zeros((2, 2)),
        feasible=lambda x: bool(x[0] >= -1e-9 and x[1] >= -1e-9 and 2.0 * x[0] + x[1] <= 8.0 + 1e-9 and x[0] + 3.0 * x[1] <= 9.0 + 1e-9 and x[0] <= 3.5 + 1e-9),
        violation=lambda x: float(max(0.0, -x[0], -x[1], 2.0 * x[0] + x[1] - 8.0, x[0] + 3.0 * x[1] - 9.0, x[0] - 3.5)),
        project=lambda x: np.array([min(max(x[0], 0.0), 3.5), max(x[1], 0.0)]),
    ),
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
        initial_point=np.array([2.8, 2.4]),
        objective=_qp_objective,
        gradient=_qp_gradient,
        hessian=_qp_hessian,
        feasible=lambda x: bool(-1.0 <= x[0] <= 3.0 and -1.0 <= x[1] <= 3.0 and x.sum() <= 3.5 + 1e-9),
        violation=lambda x: float(max(0.0, -1.0 - x[0], x[0] - 3.0, -1.0 - x[1], x[1] - 3.0, x.sum() - 3.5)),
        project=_project_qp,
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
        initial_point=np.array([1.25, 1.1]),
        objective=_portfolio_objective,
        gradient=_portfolio_gradient,
        hessian=_portfolio_hessian,
        feasible=lambda x: bool(0.0 <= x[0] <= 1.4 and 0.0 <= x[1] <= 1.4 and 0.35 - 1e-9 <= x.sum() <= 1.6 + 1e-9),
        violation=lambda x: float(max(0.0, -x[0], x[0] - 1.4, -x[1], x[1] - 1.4, 0.35 - x.sum(), x.sum() - 1.6)),
        project=_project_portfolio,
    ),
    "qcqp-intersection": ProblemDefinition(
        id="qcqp-intersection",
        type="qcqp",
        name="QCQP: Quadratic Constraint Intersection",
        description="A convex QCQP with circular and elliptical quadratic constraints.",
        dimension=2,
        variables=["x", "y"],
        compatible_methods=["interior_point", "gd", "sgd", "pgd", "newton"],
        constraints=["x^2 + y^2 <= 4", "(x + 0.8)^2 / 4 + (y - 0.2)^2 / 1.5 <= 1"],
        bounds=((-2.3, 2.3), (-2.3, 2.3)),
        initial_point=np.array([-1.2, 0.1]),
        objective=_qcqp_objective,
        gradient=_qcqp_gradient,
        hessian=_qcqp_hessian,
        feasible=lambda x: all(value <= 1e-9 for value in _qcqp_constraints(x)),
        violation=lambda x: float(max(0.0, *_qcqp_constraints(x))),
        project=_project_qcqp,
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
        objective=_qcqp_cap_objective,
        gradient=_qcqp_cap_gradient,
        hessian=_qcqp_cap_hessian,
        feasible=lambda x: bool(x[0] ** 2 + x[1] ** 2 <= 2.25 + 1e-9 and x.sum() >= -0.5 - 1e-9),
        violation=lambda x: float(max(0.0, x[0] ** 2 + x[1] ** 2 - 2.25, -0.5 - x.sum())),
        project=_project_qcqp_cap,
    ),
    "socp-cone": ProblemDefinition(
        id="socp-cone",
        type="socp",
        name="SOCP: Tilted Cone Section",
        description="A second-order cone program shown as a 2D cone cross-section.",
        dimension=2,
        variables=["x", "y"],
        compatible_methods=["interior_point"],
        constraints=["||(x, y)||_2 <= 1.85 - 0.25x + 0.1y"],
        bounds=((-2.2, 2.2), (-2.2, 2.2)),
        initial_point=np.array([0.0, 0.0]),
        objective=_socp_objective,
        gradient=lambda _: np.array([-1.0, -0.7]),
        hessian=lambda _: np.zeros((2, 2)),
        feasible=lambda x: bool(np.linalg.norm(x) <= _socp_rhs(x) + 1e-9),
        violation=lambda x: float(max(0.0, np.linalg.norm(x) - _socp_rhs(x))),
    ),
    "sdp-trace": ProblemDefinition(
        id="sdp-trace",
        type="sdp",
        name="SDP: 2x2 PSD Trace Slice",
        description="A semidefinite program visualized through X = [[a, b], [b, 1-a]].",
        dimension=2,
        variables=["a = X00", "b = X01"],
        compatible_methods=["interior_point"],
        constraints=["X is positive semidefinite", "trace(X) = 1", "X = [[a, b], [b, 1-a]]"],
        bounds=((-0.1, 1.1), (-0.65, 0.65)),
        initial_point=np.array([0.5, 0.0]),
        objective=_sdp_objective,
        gradient=lambda _: np.array([0.05, -1.0]),
        hessian=lambda _: np.zeros((2, 2)),
        feasible=_sdp_feasible,
        violation=_sdp_violation,
    ),
}


def list_problem_summaries() -> list[ProblemSummary]:
    return [problem.summary() for problem in PROBLEMS.values()]


def get_problem(problem_id: str) -> ProblemDefinition:
    if problem_id not in PROBLEMS:
        raise KeyError(f"Unknown problem id: {problem_id}")
    return PROBLEMS[problem_id]


def sample_geometry(problem_id: str, resolution: int = 80) -> GeometryResponse:
    problem = get_problem(problem_id)
    x_values = np.linspace(problem.bounds[0][0], problem.bounds[0][1], resolution)
    y_values = np.linspace(problem.bounds[1][0], problem.bounds[1][1], resolution)
    objective: list[list[float | None]] = []
    feasible: list[list[int]] = []

    for y in y_values:
        objective_row: list[float | None] = []
        feasible_row: list[int] = []
        for x in x_values:
            point = np.array([x, y])
            is_feasible = problem.feasible(point)
            feasible_row.append(1 if is_feasible else 0)
            objective_row.append(problem.objective(point) if np.isfinite(problem.objective(point)) else None)
        objective.append(objective_row)
        feasible.append(feasible_row)

    return GeometryResponse(
        problem_id=problem.id,
        problem_type=problem.type,
        x=x_values.round(5).tolist(),
        y=y_values.round(5).tolist(),
        objective=objective,
        feasible=feasible,
        boundaries=_boundaries(problem),
        annotations=_geometry_annotations(problem),
    )


def _boundaries(problem: ProblemDefinition) -> list[dict[str, object]]:
    if problem.id == "lp-basic":
        return [
            {"name": "x + y = 4", "x": [0, 4], "y": [4, 0]},
            {"name": "x = 3", "x": [3, 3], "y": [0, 2]},
            {"name": "y = 2", "x": [0, 3], "y": [2, 2]},
        ]
    if problem.id == "lp-warehouse":
        return [
            {"name": "2x + y = 8", "x": [2, 4], "y": [4, 0]},
            {"name": "x + 3y = 9", "x": [0, 4.2], "y": [3, 1.6]},
            {"name": "x = 3.5", "x": [3.5, 3.5], "y": [0, 1]},
        ]
    if problem.id == "qp-constrained":
        return [{"name": "x + y = 3.5", "x": [-1, 3], "y": [4.5, 0.5]}]
    if problem.id == "qp-portfolio":
        return [
            {"name": "x + y = 0.35", "x": [0, 0.35], "y": [0.35, 0]},
            {"name": "x + y = 1.6", "x": [0.2, 1.4], "y": [1.4, 0.2]},
        ]
    if problem.id == "qcqp-intersection":
        theta = np.linspace(0, 2 * np.pi, 160)
        return [
            {"name": "x^2 + y^2 = 4", "x": (2 * np.cos(theta)).tolist(), "y": (2 * np.sin(theta)).tolist()},
            {
                "name": "ellipse boundary",
                "x": (-0.8 + 2 * np.cos(theta)).tolist(),
                "y": (0.2 + np.sqrt(1.5) * np.sin(theta)).tolist(),
            },
        ]
    if problem.id == "qcqp-disk-cap":
        theta = np.linspace(0, 2 * np.pi, 160)
        return [
            {"name": "x^2 + y^2 = 2.25", "x": (1.5 * np.cos(theta)).tolist(), "y": (1.5 * np.sin(theta)).tolist()},
            {"name": "x + y = -0.5", "x": [(-1 - np.sqrt(17)) / 4, (-1 + np.sqrt(17)) / 4], "y": [(-1 + np.sqrt(17)) / 4, (-1 - np.sqrt(17)) / 4]},
        ]
    if problem.id == "sdp-trace":
        a = np.linspace(0, 1, 160)
        radius = np.sqrt(np.maximum(a * (1 - a), 0))
        return [
            {"name": "PSD upper boundary", "x": a.tolist(), "y": radius.tolist()},
            {"name": "PSD lower boundary", "x": a.tolist(), "y": (-radius).tolist()},
        ]
    return []


def _geometry_annotations(problem: ProblemDefinition) -> list[str]:
    if problem.type == "sdp":
        return ["The 2D plot uses a trace-one slice of the PSD cone, not the full matrix cone."]
    if problem.type == "socp":
        return ["The feasible shading is a 2D projection of the second-order cone inequality."]
    return []


def build_cvxpy_problem(problem: ProblemDefinition):
    if problem.type == "lp":
        x = cp.Variable(2, name="x")
        c, a_ub, b_ub, bounds = scipy_lp_data(problem)
        objective = cp.Minimize(c @ x)
        constraints = [a_ub @ x <= b_ub]
        for index, (lower, upper) in enumerate(bounds):
            if lower is not None:
                constraints.append(x[index] >= lower)
            if upper is not None:
                constraints.append(x[index] <= upper)
        return cp.Problem(objective, constraints), {"x": x}

    if problem.id == "qp-constrained":
        x = cp.Variable(2, name="x")
        q_mat = np.array([[3.0, 0.6], [0.6, 1.4]])
        q_vec = np.array([-2.0, -1.0])
        objective = cp.Minimize(0.5 * cp.quad_form(x, q_mat) + q_vec @ x)
        constraints = [x >= -1, x <= 3, x[0] + x[1] <= 3.5]
        return cp.Problem(objective, constraints), {"x": x}

    if problem.id == "qp-portfolio":
        x = cp.Variable(2, name="x")
        q_mat = np.array([[2.4, 0.8], [0.8, 1.7]])
        q_vec = np.array([-1.1, -1.5])
        objective = cp.Minimize(0.5 * cp.quad_form(x, q_mat) + q_vec @ x)
        constraints = [x >= 0, x <= 1.4, x[0] + x[1] >= 0.35, x[0] + x[1] <= 1.6]
        return cp.Problem(objective, constraints), {"x": x}

    if problem.id == "qcqp-intersection":
        x = cp.Variable(2, name="x")
        objective = cp.Minimize(cp.sum_squares(x - np.array([1.4, 0.35])))
        constraints = [
            cp.sum_squares(x) <= 4,
            cp.square(x[0] + 0.8) / 4 + cp.square(x[1] - 0.2) / 1.5 <= 1,
        ]
        return cp.Problem(objective, constraints), {"x": x}

    if problem.id == "qcqp-disk-cap":
        x = cp.Variable(2, name="x")
        objective = cp.Minimize(cp.square(x[0] - 0.6) + 0.6 * cp.square(x[1] + 0.8))
        constraints = [cp.sum_squares(x) <= 2.25, x[0] + x[1] >= -0.5]
        return cp.Problem(objective, constraints), {"x": x}

    if problem.id == "socp-cone":
        x = cp.Variable(2, name="x")
        rhs = 1.85 - 0.25 * x[0] + 0.1 * x[1]
        objective = cp.Minimize(-x[0] - 0.7 * x[1])
        constraints = [cp.SOC(rhs, x)]
        return cp.Problem(objective, constraints), {"x": x}

    if problem.id == "sdp-trace":
        x = cp.Variable((2, 2), symmetric=True, name="X")
        objective = cp.Minimize(-x[0, 1] + 0.05 * x[0, 0])
        constraints = [x >> 0, cp.trace(x) == 1]
        return cp.Problem(objective, constraints), {"X": x}

    raise KeyError(f"No CVXPY builder for {problem.id}")


def scipy_lp_data(problem: ProblemDefinition):
    if problem.id == "lp-basic":
        c = np.array([-3.0, -2.0])
        a_ub = np.array([[1.0, 1.0]])
        b_ub = np.array([4.0])
        bounds = [(0.0, 3.0), (0.0, 2.0)]
        return c, a_ub, b_ub, bounds

    if problem.id == "lp-warehouse":
        c = np.array([-4.0, -3.0])
        a_ub = np.array([[2.0, 1.0], [1.0, 3.0]])
        b_ub = np.array([8.0, 9.0])
        bounds = [(0.0, 3.5), (0.0, None)]
        return c, a_ub, b_ub, bounds

    raise KeyError(f"No SciPy LP data for {problem.id}")
