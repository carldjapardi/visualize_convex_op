from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schemas import Iterate


@dataclass
class SolveResult:
    status: str
    solver: str
    message: str
    objective_value: float | None = None
    variables: dict[str, Any] = field(default_factory=dict)
    iterations: int = 0
    trace: list[Iterate] = field(default_factory=list)
    history: dict[str, list[float]] = field(default_factory=dict)
