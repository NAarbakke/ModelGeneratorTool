"""Planar measurement math: exact 4-point homography and point mapping.

Pure numpy - no OpenCV imports, so the smoke test runs without a display.
"""

from __future__ import annotations

import numpy as np

# Reference rectangle presets: (width, height) in mm, as seen with the longer
# side horizontal or vertical - orientation doesn't matter, only the corner order.
REFERENCE_PRESETS: dict[str, tuple[float, float]] = {
    "a4": (210.0, 297.0),
    "letter": (215.9, 279.4),
    "credit-card": (85.6, 53.98),
}


def rect_corners(width_mm: float, height_mm: float) -> np.ndarray:
    """Metric corners of the reference rectangle in click order: TL, TR, BR, BL."""
    return np.array(
        [[0.0, 0.0], [width_mm, 0.0], [width_mm, height_mm], [0.0, height_mm]], dtype=float
    )


def solve_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Exact 3x3 homography mapping 4 src points onto 4 dst points (DLT).

    Raises numpy.linalg.LinAlgError for degenerate (e.g. collinear) corners.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    rows, rhs = [], []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.append(v)
    h = np.linalg.solve(np.array(rows), np.array(rhs))
    return np.append(h, 1.0).reshape(3, 3)


def apply_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map Nx2 points through H (homogeneous divide)."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    homogeneous = np.hstack([pts, np.ones((len(pts), 1))])
    mapped = homogeneous @ H.T
    return mapped[:, :2] / mapped[:, 2:3]
