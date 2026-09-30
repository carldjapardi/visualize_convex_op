from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .compatibility import validate_compatible_methods
from .schemas import MethodId, ProblemSummary, ProblemType

ArrayFn = Callable[[np.ndarray], float]
BoundaryFn = Callable[[], list[dict[str, Any]]]


@dataclass(frozen=True)
class LPData:
    """Standard form for 2D LP solvers: min c^T x s.t. A_ub x <= b_ub and box bounds."""

    c: np.ndarray
    a_ub: np.ndarray
    b_ub: np.ndarray
    bounds: list[tuple[float | None, float | None]]


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
    feasible: Callable[[np.ndarray], bool]
    violation: Callable[[np.ndarray], float]
    gradient: Callable[[np.ndarray], np.ndarray] | None = None
    hessian: Callable[[np.ndarray], np.ndarray] | None = None
    project: Callable[[np.ndarray], np.ndarray] | None = None
    boundaries_fn: BoundaryFn | None = None
    annotations: list[str] = field(default_factory=list)
    default_step_size: float = 0.12
    default_max_iterations: int = 60
    lp_data: LPData | None = None
    trace_message: str | None = None
    objective_expression: str = ""
    learning_goal: str = ""

    def __post_init__(self) -> None:
        validate_compatible_methods(self.type, self.compatible_methods)

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
            objective_expression=self.objective_expression,
            learning_goal=self.learning_goal,
            initial_point=self.initial_point.tolist(),
            plot_bounds=[list(axis) for axis in self.bounds],
            default_step_size=self.default_step_size,
            default_max_iterations=self.default_max_iterations,
        )

    def boundaries(self) -> list[dict[str, Any]]:
        if self.boundaries_fn is None:
            return []
        return self.boundaries_fn()
