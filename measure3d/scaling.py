"""Scale calibration and measurement math, plus the scale.json sidecar.

Pure numpy/stdlib - no Open3D or pycolmap imports, so the smoke test can run
without the heavy dependencies.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 3D points."""
    return float(np.linalg.norm(np.asarray(b, dtype=float) - np.asarray(a, dtype=float)))


def compute_scale(known_distance_mm: float, picked_distance_units: float) -> float:
    """mm per reconstruction unit, from a picked point pair of known real length."""
    if known_distance_mm <= 0:
        raise ValueError("known distance must be positive (mm)")
    if picked_distance_units <= 0:
        raise ValueError("picked points coincide - pick two distinct points")
    return known_distance_mm / picked_distance_units


def pair_distances_mm(
    points: np.ndarray, indices: Sequence[int], scale_mm_per_unit: float
) -> list[float]:
    """Real-world distance for each consecutive pair of picked point indices.

    Indices are chunked as (0,1), (2,3), ...; a trailing unpaired index is ignored.
    """
    pts = np.asarray(points, dtype=float)
    return [
        distance(pts[indices[i]], pts[indices[i + 1]]) * scale_mm_per_unit
        for i in range(0, len(indices) - 1, 2)
    ]


def scaled_aabb_mm(
    points: np.ndarray, scale_mm_per_unit: float
) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box (min, max) of the point set, in mm."""
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        raise ValueError("no points")
    return pts.min(axis=0) * scale_mm_per_unit, pts.max(axis=0) * scale_mm_per_unit


def save_scale(
    path: Path,
    scale_mm_per_unit: float,
    known_distance_mm: float,
    picked_distance_units: float,
) -> None:
    payload = {
        "scale_mm_per_unit": scale_mm_per_unit,
        "known_distance_mm": known_distance_mm,
        "picked_distance_units": picked_distance_units,
        "units": "mm",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_scale(path: Path) -> float | None:
    """Return the stored scale factor (mm per unit), or None if not calibrated."""
    if not path.exists():
        return None
    return float(json.loads(path.read_text())["scale_mm_per_unit"])
