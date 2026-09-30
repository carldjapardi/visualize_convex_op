from __future__ import annotations

from dataclasses import replace

from ..problem_definition import ProblemDefinition
from .lp import build_lp_problems
from .nonconvex import build_nonconvex_problems
from .qcqp import build_qcqp_problems
from .qp import build_qp_problems
from .sdp import build_sdp_problems
from .socp import build_socp_problems

GUIDES: dict[str, tuple[str, str]] = {
    "lp-basic": ("minimize −3x − 2y", "A linear objective reaches its best value at a feasible corner."),
    "lp-ratio-blend": ("minimize −4x − 2y", "A required blend ratio creates a slanted edge that can set the optimum."),
    "lp-diet": ("minimize 2x + 1.5y", "Minimum nutrient requirements keep the least-cost solution away from zero."),
    "lp-delivery-window": ("minimize 2x + 4y", "A lower and upper delivery target create a feasible strip rather than a corner near zero."),
    "lp-alternate-optima": ("minimize −x − y", "Every feasible point on one boundary segment has the same optimal objective."),
    "qp-constrained": ("minimize 1.5x² + 0.6xy + 0.7y² − 2x − y", "Compare curved objective contours with a flat feasible boundary."),
    "qp-portfolio": ("minimize 1.2x² + 0.8xy + 0.85y² − 1.1x − 1.5y", "The quadratic risk term changes the best allocation."),
    "qp-least-squares": ("minimize 2x² + 1.25y² + 3x − 0.5y", "Gradient descent leaves the box; projection keeps the constrained path inside."),
    "qp-elastic-net-slice": ("minimize 2.5x² + 0.6y² − 1.2x − 0.8y", "Different curvature along each axis changes the descent path."),
    "qp-saddle-slice": ("minimize x² + 0.3xy + 1.5y² − x − 1.2y", "Newton's method uses curvature to approach the bowl minimum quickly."),
    "qcqp-intersection": ("minimize (x − 1.4)² + (y − 0.35)²", "Compare GD leaving the disk–ellipse intersection with PGD staying inside."),
    "qcqp-disk-cap": ("minimize (x − 0.6)² + 0.6(y + 0.8)²", "A linear cut changes the optimum on a quadratic disk."),
    "qcqp-trust-region": ("minimize −2x − y", "A linear objective can still be a QCQP when the constraint is quadratic."),
    "qcqp-ellipse-box": ("minimize (x − 1.8)² + (y − 1.6)²", "Projection keeps steps inside an ellipse and a box."),
    "socp-cone": ("minimize −x − 0.7y", "A norm bound creates a curved feasible boundary."),
    "socp-hyperbolic": ("minimize −1.2x − 0.4y", "A steep affine bound makes an asymmetric cone section."),
    "socp-norm-ball-intersect": ("minimize −x − 0.5y", "A box clips the round norm-ball feasible set."),
    "sdp-trace": ("minimize 0.05a − b", "Positive semidefiniteness becomes a curved region in a trace-one matrix slice."),
    "sdp-max-cut-slice": ("minimize −0.3a − 0.8b", "Changing a linear matrix objective moves the optimum along the PSD boundary."),
    "sdp-dual-slice": ("minimize 0.1a + 0.6b", "Eigenvalues show whether the resulting matrix stays positive semidefinite."),
    "nonconvex-double-well": ("minimize (x² − 1)² + 0.35y² + 0.25x", "Starting in the right valley can lead to a local solution above the left valley."),
    "nonconvex-himmelblau": ("minimize (x² + y − 11)² + (x + y² − 7)²", "Four separated basins show why the starting point matters."),
    "nonconvex-rippled-bowl": ("minimize 0.18(x² + y²) + 2 − cos(3x) − cos(3y) + 0.12x", "Small ripples can trap descent before it reaches the broad bowl's lowest area."),
}


def build_all_problems() -> dict[str, ProblemDefinition]:
    problems: dict[str, ProblemDefinition] = {}
    for builder in (
        build_lp_problems,
        build_qp_problems,
        build_qcqp_problems,
        build_socp_problems,
        build_sdp_problems,
        build_nonconvex_problems,
    ):
        batch = builder()
        overlap = set(problems) & set(batch)
        if overlap:
            raise ValueError(f"Duplicate problem ids: {overlap}")
        problems.update(batch)

    if set(problems) != set(GUIDES):
        raise ValueError("Every problem needs a matching objective expression and learning goal")
    return {
        problem_id: replace(problem, objective_expression=GUIDES[problem_id][0], learning_goal=GUIDES[problem_id][1])
        for problem_id, problem in problems.items()
    }
