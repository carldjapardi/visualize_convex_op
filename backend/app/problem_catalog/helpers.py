from __future__ import annotations

import numpy as np


def quad_form(q_mat: np.ndarray, q_vec: np.ndarray, x: np.ndarray) -> float:
    return float(0.5 * x @ q_mat @ x + q_vec @ x)


def quad_gradient(q_mat: np.ndarray, q_vec: np.ndarray, x: np.ndarray) -> np.ndarray:
    return q_mat @ x + q_vec


def project_box(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.clip(x, lower, upper)


def project_halfspace_sum(
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    sum_lower: float | None,
    sum_upper: float | None,
) -> np.ndarray:
    point = project_box(x, lower, upper)
    if sum_upper is not None and point.sum() > sum_upper:
        point = point - ((point.sum() - sum_upper) / 2.0) * np.ones(2)
    if sum_lower is not None and point.sum() < sum_lower:
        point = point + ((sum_lower - point.sum()) / 2.0) * np.ones(2)
    return np.clip(point, lower, upper)


def circle_boundary(radius: float, n: int = 160) -> dict[str, object]:
    theta = np.linspace(0, 2 * np.pi, n)
    return {
        "name": f"x^2 + y^2 = {radius ** 2}",
        "x": (radius * np.cos(theta)).tolist(),
        "y": (radius * np.sin(theta)).tolist(),
    }


def ellipse_boundary(cx: float, cy: float, rx: float, ry: float, n: int = 160) -> dict[str, object]:
    theta = np.linspace(0, 2 * np.pi, n)
    return {
        "name": "ellipse boundary",
        "x": (cx + rx * np.cos(theta)).tolist(),
        "y": (cy + ry * np.sin(theta)).tolist(),
    }


def psd_slice_boundaries(n: int = 160) -> list[dict[str, object]]:
    a = np.linspace(0, 1, n)
    radius = np.sqrt(np.maximum(a * (1 - a), 0))
    return [
        {"name": "PSD upper boundary", "x": a.tolist(), "y": radius.tolist()},
        {"name": "PSD lower boundary", "x": a.tolist(), "y": (-radius).tolist()},
    ]


def sdp_feasible(a: float, b: float) -> bool:
    return 0.0 <= a <= 1.0 and b * b <= a * (1.0 - a) + 1e-9


def sdp_violation(a: float, b: float) -> float:
    return float(max(0.0, -a, a - 1.0, b * b - a * (1.0 - a)))
