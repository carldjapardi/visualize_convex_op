from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ProblemType = Literal["lp", "qp", "qcqp", "socp", "sdp", "nonconvex"]
MethodId = Literal["simplex", "interior_point", "gd", "sgd", "pgd", "newton"]


class ProblemSummary(BaseModel):
    id: str
    type: ProblemType
    name: str
    description: str
    dimension: int
    variables: list[str]
    compatible_methods: list[MethodId]
    constraints: list[str]
    objective_expression: str
    learning_goal: str
    initial_point: list[float]
    plot_bounds: list[list[float]]
    default_step_size: float
    default_max_iterations: int


class MethodSummary(BaseModel):
    id: MethodId
    name: str
    description: str
    best_for: list[ProblemType]


class SolveRequest(BaseModel):
    problem_id: str = Field(default="lp-basic")
    method: MethodId = Field(default="interior_point")
    max_iterations: int = Field(default=60, ge=1, le=500)
    step_size: float = Field(default=0.12, gt=0, le=2)
    seed: int = Field(default=7, ge=0)
    start_point: tuple[float, float] | None = None


class GeometryRequest(BaseModel):
    problem_id: str = Field(default="lp-basic")
    resolution: int = Field(default=80, ge=20, le=180)


class Iterate(BaseModel):
    iteration: int
    x: list[float]
    objective: float
    violation: float = 0.0
    step_norm: float = 0.0


class GeometryResponse(BaseModel):
    problem_id: str
    problem_type: ProblemType
    x: list[float]
    y: list[float]
    objective: list[list[float | None]]
    feasible: list[list[int]]
    boundaries: list[dict[str, Any]] = Field(default_factory=list)
    annotations: list[str] = Field(default_factory=list)


class SolveResponse(BaseModel):
    problem: ProblemSummary
    method: MethodSummary
    status: str
    compatible: bool
    solver: str | None = None
    message: str
    objective_value: float | None = None
    variables: dict[str, Any] = Field(default_factory=dict)
    iterations: int = 0
    trace: list[Iterate] = Field(default_factory=list)
    history: dict[str, list[float]] = Field(default_factory=dict)
    geometry: GeometryResponse | None = None
