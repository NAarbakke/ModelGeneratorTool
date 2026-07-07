"""Structure-from-motion via pycolmap: feature extraction → matching → mapping."""

from __future__ import annotations

import shutil
from dataclasses import dataclass

import numpy as np

from .project import Workspace


@dataclass
class SfmSummary:
    total_images: int
    registered_images: int
    num_points: int
    mean_reprojection_error: float


def run_sfm(ws: Workspace) -> SfmSummary:
    """Run sparse reconstruction; the best model is written to <photos>/.measure3d/sparse/0."""
    import pycolmap

    images = ws.list_images()
    if len(images) < 3:
        raise RuntimeError(
            f"found only {len(images)} image(s) in {ws.photos_dir} - SfM needs at least 3 "
            "overlapping photos (10-30 from different angles recommended)"
        )

    # Fresh run: stale database/models would mix with the new mapping.
    if ws.database_path.exists():
        ws.database_path.unlink()
    if ws.sparse_dir.exists():
        shutil.rmtree(ws.sparse_dir)
    ws.sparse_dir.mkdir(parents=True)

    print(f"[1/3] Extracting features from {len(images)} images...")
    pycolmap.extract_features(ws.database_path, ws.photos_dir)
    print("[2/3] Matching features (exhaustive)...")
    pycolmap.match_exhaustive(ws.database_path)
    print("[3/3] Mapping (incremental SfM)...")
    maps = pycolmap.incremental_mapping(ws.database_path, ws.photos_dir, ws.sparse_dir)
    if not maps:
        raise RuntimeError(
            "reconstruction failed: no images could be registered - add more overlapping "
            "photos or improve texture/lighting"
        )

    best = max(maps.values(), key=lambda rec: rec.num_reg_images())
    ws.sparse_model_dir.mkdir(parents=True, exist_ok=True)
    best.write(ws.sparse_model_dir)

    return SfmSummary(
        total_images=len(images),
        registered_images=best.num_reg_images(),
        num_points=best.num_points3D(),
        mean_reprojection_error=float(best.compute_mean_reprojection_error()),
    )


def load_sparse_points(ws: Workspace) -> tuple[np.ndarray, np.ndarray]:
    """Sparse 3D points and their RGB colors (0-1) for the interactive viewer."""
    import pycolmap

    if not ws.has_sparse():
        raise RuntimeError(
            f"no reconstruction found in {ws.sparse_model_dir} - run `measure3d reconstruct` first"
        )
    rec = pycolmap.Reconstruction(ws.sparse_model_dir)
    pts3d = rec.points3D.values()
    points = np.array([p.xyz for p in pts3d], dtype=float)
    colors = np.array([p.color for p in pts3d], dtype=float) / 255.0
    return points, colors
