"""Scale calibration and measurement math, plus the scale.json calibration sidecar.

Every tool stores its calibration in the same file, <images>/.measure3d/scale.json,
as a 4x4 similarity transform from the SfM world to millimeters:
  - method "points": two picked points of known distance (scale only, SfM axes);
  - method "charuco/...": a ChArUco board (scale + board frame: origin at the
    board center, Z up).

Pure numpy/stdlib - no Open3D or pycolmap imports, so the smoke test can run
without the heavy dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from .geometry import apply_transform, scale_of


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


@dataclass
class Calibration:
    """SfM world -> mm. `board_frame` means the axes are the ChArUco board's (Z up)."""

    world_to_mm: np.ndarray
    method: str
    info: dict = field(default_factory=dict)

    @property
    def scale_mm_per_unit(self) -> float:
        return scale_of(self.world_to_mm)

    @property
    def board_frame(self) -> bool:
        return self.method.startswith("charuco")

    def to_mm(self, points: np.ndarray) -> np.ndarray:
        return apply_transform(self.world_to_mm, points)


def save_calibration(path: Path, world_to_mm: np.ndarray, method: str, **info) -> None:
    T = np.asarray(world_to_mm, dtype=float)
    payload = {
        "world_to_mm": T.tolist(),
        "scale_mm_per_unit": scale_of(T),
        "method": method,
        "units": "mm",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **info,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_calibration(path: Path) -> Calibration | None:
    """The stored calibration, or None if not calibrated. Files without a transform
    (older picked-points calibrations) load as a pure scale."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if "world_to_mm" in data:
        T = np.array(data["world_to_mm"], dtype=float)
    else:
        T = np.diag([float(data["scale_mm_per_unit"])] * 3 + [1.0])
    info = {k: v for k, v in data.items()
            if k not in ("world_to_mm", "scale_mm_per_unit", "method", "units", "created")}
    return Calibration(T, data.get("method", "points"), info)


def save_scale(
    path: Path,
    scale_mm_per_unit: float,
    known_distance_mm: float,
    picked_distance_units: float,
) -> None:
    """Picked-points calibration: a pure scale, SfM axes unchanged."""
    save_calibration(
        path, np.diag([scale_mm_per_unit] * 3 + [1.0]), "points",
        known_distance_mm=known_distance_mm, picked_distance_units=picked_distance_units,
    )


def load_scale(path: Path) -> float | None:
    """Return the stored scale factor (mm per unit), or None if not calibrated."""
    cal = load_calibration(path)
    return None if cal is None else cal.scale_mm_per_unit
