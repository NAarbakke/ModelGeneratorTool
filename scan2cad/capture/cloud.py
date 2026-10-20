"""Isolate the selected object as an oriented point cloud in mm (board frame).

Backends:
  dense   OpenMVS (CPU) or COLMAP patch-match (CUDA) depth maps via core.dense,
          then a mask vote: a point survives if it lands inside the SAM 2 mask
          in most of the views that see it.
  hull    visual hull: carve a voxel grid with the masks, marching cubes for a
          sub-voxel surface. Needs only poses + masks (no GPU, no OpenMVS), but
          cannot recover concavities that no silhouette reveals.
  sparse  the SfM points themselves, mask-voted. Quick look only.

SfM always runs on the unmasked frames: background texture (and the board) is
what pins down the camera poses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.geometry import apply_transform, scale_of, transform_normals

from ..workspace import ScanWorkspace

VOTE_FRACTION = 0.8      # dense/sparse: share of seeing views where the point is in the mask
HULL_FRACTION = 0.95     # hull: tolerate a few bad masks per voxel
MIN_VIEWS = 3
HULL_RESOLUTION = 192    # voxels along the longest side
FLOOR_CUT_MM = 0.75      # drop points this close to the board plane (contact zone)


@dataclass
class View:
    """One registered frame: projection function inputs plus its mask."""

    camera: object          # pycolmap.Camera
    cam_from_world: np.ndarray  # 3x4
    mask: np.ndarray        # bool (h, w)

    @property
    def center(self) -> np.ndarray:
        R, t = self.cam_from_world[:, :3], self.cam_from_world[:, 3]
        return -R.T @ t

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixel coords (n, 2), COLMAP convention, and a front-facing flag."""
        cam = points @ self.cam_from_world[:, :3].T + self.cam_from_world[:, 3]
        front = cam[:, 2] > 1e-9
        uv = np.full((len(points), 2), -1.0)
        if front.any():
            uv[front] = self.camera.img_from_cam(cam[front])
        return uv, front


def load_views(ws: ScanWorkspace) -> tuple[list[View], object]:
    """Registered frames that have a mask, plus the pycolmap reconstruction."""
    import cv2
    import pycolmap

    if not ws.sfm.has_sparse():
        raise RuntimeError("no SfM model - run `scan2cad reconstruct` first")
    rec = pycolmap.Reconstruction(ws.sfm.sparse_model_dir)
    views = []
    for image in rec.images.values():
        if not image.has_pose:
            continue
        mpath = ws.mask_path(image.name)
        if not mpath.exists():
            continue
        mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE) > 127
        views.append(View(image.camera, image.cam_from_world().matrix(), mask))
    if len(views) < MIN_VIEWS:
        raise RuntimeError(f"only {len(views)} registered frames have masks - run `scan2cad mask`")
    return views, rec


def mask_votes(points: np.ndarray, views: list[View], chunk: int = 500_000
               ) -> tuple[np.ndarray, np.ndarray]:
    """Per point: number of views it is visible in (in frame), and how many put it in the mask."""
    seen = np.zeros(len(points), dtype=np.int32)
    inside = np.zeros(len(points), dtype=np.int32)
    for s in range(0, len(points), chunk):
        pts = points[s:s + chunk]
        for v in views:
            h, w = v.mask.shape
            uv, front = v.project(pts)
            col, row = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
            ok = front & (col >= 0) & (col < w) & (row >= 0) & (row < h)
            seen[s:s + chunk] += ok
            hit = np.zeros(len(pts), dtype=bool)
            hit[ok] = v.mask[row[ok], col[ok]]
            inside[s:s + chunk] += hit
    return seen, inside


def vote_filter(points: np.ndarray, views: list[View], fraction: float = VOTE_FRACTION,
                min_views: int = MIN_VIEWS) -> np.ndarray:
    seen, inside = mask_votes(points, views)
    return (seen >= min_views) & (inside >= fraction * np.maximum(seen, 1))


def visual_hull(views: list[View], bounds: tuple[np.ndarray, np.ndarray],
                resolution: int = HULL_RESOLUTION, fraction: float = HULL_FRACTION
                ) -> tuple[np.ndarray, np.ndarray, float]:
    """Carve a voxel grid; return surface points, outward normals, voxel size (world units)."""
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes

    lo, hi = bounds
    voxel = float(np.max(hi - lo)) / resolution
    dims = np.maximum(np.ceil((hi - lo) / voxel).astype(int), 2) + 2
    origin = lo - voxel
    idx = np.indices(dims).reshape(3, -1).T
    centers = origin + (idx + 0.5) * voxel
    seen, inside = mask_votes(centers, views)
    occ = (seen >= MIN_VIEWS) & (inside >= fraction * np.maximum(seen, 1))
    grid = gaussian_filter(occ.reshape(dims).astype(np.float32), sigma=0.8)
    if grid.max() < 0.5:
        raise RuntimeError("visual hull is empty - masks and camera poses disagree")
    verts, _, normals, _ = marching_cubes(grid, level=0.5)
    points = origin + (verts + 0.5) * voxel
    return points, normals, voxel  # skimage normals already point toward lower occupancy


def build_object_cloud(ws: ScanWorkspace, backend: str = "hull", openmvs_dir: str | None = None,
                       resolution: int = HULL_RESOLUTION, floor_cut_mm: float | None = FLOOR_CUT_MM,
                       dense_backend: str = "auto") -> dict:
    """Run the chosen backend and write ws.object_ply (mm, board frame, with normals)."""
    import open3d as o3d

    cal = ws.calibration()
    if cal is None:
        raise RuntimeError("no scale - run `scan2cad align` (ChArUco) or "
                           "`measure3d calibrate <frames>` (known distance) first")
    T = cal.world_to_mm
    views, rec = load_views(ws)
    sparse = np.array([p.xyz for p in rec.points3D.values()], dtype=float)
    obj_sparse = sparse[vote_filter(sparse, views)] if len(sparse) else sparse

    if backend == "hull":
        if len(obj_sparse) < 10:
            raise RuntimeError("too few SfM points on the object to bound the hull")
        lo, hi = np.percentile(obj_sparse, 1, axis=0), np.percentile(obj_sparse, 99, axis=0)
        pad = 0.25 * (hi - lo) + 1e-6
        pts, nrm, voxel = visual_hull(views, (lo - pad, hi + pad), resolution)
        info = {"voxel_mm": voxel * scale_of(T)}
    elif backend == "dense":
        from core import dense

        b = dense.resolve_backend(dense_backend, openmvs_dir)
        pcd = o3d.io.read_point_cloud(str(dense.densify(ws.sfm, b, openmvs_dir)))
        pts = np.asarray(pcd.points)
        keep = vote_filter(pts, views)
        nrm = np.asarray(pcd.normals)[keep] if pcd.has_normals() else None
        pts = pts[keep]
        info = {"dense_points": len(keep)}
    elif backend == "sparse":
        pts, nrm, info = obj_sparse, None, {}
    else:
        raise ValueError(f"unknown backend {backend!r}")

    pts_mm = apply_transform(T, pts)
    nrm_mm = transform_normals(T, nrm) if nrm is not None else None
    if floor_cut_mm is not None and cal.board_frame:
        keep = pts_mm[:, 2] > floor_cut_mm  # board frame: z = 0 is the board surface
        pts_mm = pts_mm[keep]
        nrm_mm = nrm_mm[keep] if nrm_mm is not None else None

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_mm))
    if nrm_mm is not None:
        pcd.normals = o3d.utility.Vector3dVector(nrm_mm)
    pcd = clean(pcd)
    if not pcd.has_normals():
        orient_normals(pcd, np.array([apply_transform(T, v.center[None])[0] for v in views]))
    o3d.io.write_point_cloud(str(ws.object_ply), pcd)
    ext = np.ptp(np.asarray(pcd.points), axis=0)
    return {"backend": backend, "points": len(pcd.points), "views": len(views),
            "extent_mm": [round(float(e), 2) for e in ext], **info}


def clean(pcd, nb_neighbors: int = 20, std_ratio: float = 2.5):
    """Statistical outlier removal, then keep the largest connected cluster."""
    import open3d as o3d

    if len(pcd.points) < 50:
        return pcd
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    spacing = float(np.median(pcd.compute_nearest_neighbor_distance()))
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        labels = np.asarray(pcd.cluster_dbscan(eps=4 * spacing, min_points=5))
    if labels.max() < 0:
        return pcd
    largest = np.bincount(labels[labels >= 0]).argmax()
    return pcd.select_by_index(np.flatnonzero(labels == largest))


def orient_normals(pcd, camera_centers: np.ndarray) -> None:
    """Estimate normals and point each toward the nearest camera (outward for a scanned object)."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    spacing = float(np.median(pcd.compute_nearest_neighbor_distance()))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * spacing, max_nn=30))
    pts, nrm = np.asarray(pcd.points), np.asarray(pcd.normals)
    _, nearest = cKDTree(camera_centers).query(pts)
    flip = np.sum((camera_centers[nearest] - pts) * nrm, axis=1) < 0
    nrm[flip] *= -1
    pcd.normals = o3d.utility.Vector3dVector(nrm)


def load_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(path))
    if not len(pcd.points):
        raise RuntimeError(f"empty point cloud: {path}")
    if not pcd.has_normals():
        raise RuntimeError(f"{path} has no normals")
    return np.asarray(pcd.points), np.asarray(pcd.normals)
