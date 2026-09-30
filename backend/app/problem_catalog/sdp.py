from __future__ import annotations

import numpy as np

from ..problem_definition import ProblemDefinition
from .helpers import psd_slice_boundaries, sdp_feasible, sdp_violation


def _sdp_objective(coef_a: float, coef_b: float):
    return lambda x: float(coef_a * x[0] + coef_b * x[1])


def _make_sdp(
    problem_id: str,
    name: str,
    description: str,
    coef_a: float,
    coef_b: float,
    initial_point: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> ProblemDefinition:
    return ProblemDefinition(
        id=problem_id,
        type="sdp",
        name=name,
        description=description,
        dimension=2,
        variables=["a = X00", "b = X01"],
        compatible_methods=["interior_point"],
        constraints=["X is positive semidefinite", "trace(X) = 1", "X = [[a, b], [b, 1-a]]"],
        bounds=bounds,
        initial_point=initial_point,
        objective=_sdp_objective(coef_a, coef_b),
        gradient=lambda _: np.array([coef_a, coef_b]),
        hessian=lambda _: np.zeros((2, 2)),
        feasible=lambda x: sdp_feasible(x[0], x[1]),
        violation=lambda x: sdp_violation(x[0], x[1]),
        boundaries_fn=psd_slice_boundaries,
        annotations=["The domain is a trace-one slice of 2×2 PSD matrices; 3D height is objective value."],
    )


def build_sdp_problems() -> dict[str, ProblemDefinition]:
    return {
        "sdp-trace": _make_sdp(
            "sdp-trace",
            "SDP: 2x2 PSD Trace Slice",
            "A semidefinite program visualized through X = [[a, b], [b, 1-a]].",
            0.05,
            -1.0,
            np.array([0.5, 0.0]),
            ((-0.1, 1.1), (-0.65, 0.65)),
        ),
        "sdp-max-cut-slice": _make_sdp(
            "sdp-max-cut-slice",
            "SDP: Objective Tilt",
            "A different linear objective on the same trace-one PSD slice.",
            -0.3,
            -0.8,
            np.array([0.25, 0.4]),
            ((-0.1, 1.1), (-0.65, 0.65)),
        ),
        "sdp-dual-slice": _make_sdp(
            "sdp-dual-slice",
            "SDP: Eigenvalue Check",
            "Inspect eigenvalue diagnostics after solving on a trace-one PSD slice.",
            0.1,
            0.6,
            np.array([0.7, 0.1]),
            ((-0.1, 1.1), (-0.65, 0.65)),
        ),
        "sdp-narrow-cone": _make_sdp(
            "sdp-narrow-cone",
            "SDP: Boundary Zoom",
            "A close view of the boundary of the same trace-one PSD slice.",
            0.0,
            -1.2,
            np.array([0.02, 0.0]),
            ((-0.05, 0.35), (-0.25, 0.25)),
        ),
        "sdp-wide-cone": _make_sdp(
            "sdp-wide-cone",
            "SDP: Interior Start",
            "A different linear objective starting from the center of the PSD slice.",
            -0.2,
            0.4,
            np.array([0.5, 0.0]),
            ((-0.1, 1.1), (-0.65, 0.65)),
        ),
    }
