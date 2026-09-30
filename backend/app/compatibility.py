from __future__ import annotations

from .schemas import MethodId, ProblemType

TYPE_ALLOWED_METHODS: dict[ProblemType, frozenset[MethodId]] = {
    "lp": frozenset({"simplex", "interior_point"}),
    "qp": frozenset({"interior_point", "gd", "sgd", "pgd", "newton"}),
    "qcqp": frozenset({"interior_point", "gd", "sgd", "pgd", "newton"}),
    "socp": frozenset({"interior_point"}),
    "sdp": frozenset({"interior_point"}),
    "nonconvex": frozenset({"gd", "sgd", "newton"}),
}


def allowed_methods(problem_type: ProblemType) -> frozenset[MethodId]:
    return TYPE_ALLOWED_METHODS[problem_type]


def validate_compatible_methods(problem_type: ProblemType, compatible_methods: list[MethodId]) -> None:
    allowed = allowed_methods(problem_type)
    invalid = [method for method in compatible_methods if method not in allowed]
    if invalid:
        raise ValueError(
            f"Methods {invalid} are not allowed for problem type {problem_type}. Allowed: {sorted(allowed)}"
        )
