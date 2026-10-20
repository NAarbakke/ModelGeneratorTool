"""Score a CAD candidate against the scan. All distances in mm.

  scan -> CAD   does the model explain every scanned point? (missing features)
  CAD -> scan   is every *observable* bit of model surface backed by scan points?
                (invented features). "Observable" = seen unoccluded by >= 3 cameras
                within 60 degrees of head-on, roughly where multi-view stereo can
                reconstruct. The bottom on the board and the walls of deep holes
                (seen only at grazing angles) are excluded, otherwise every model
                would be penalized for the scan's blind spots.
  f_score       harmonic mean of both directions' shares within
                max(F_SCORE_FACTOR x tolerance, SPACING_FACTOR x point spacing).
                At 1x tolerance (point noise) even a perfect model scores low,
                because CAD -> scan distances are bounded by point spacing, not noise:
                for points spread with median spacing s, P(nearest > r) = 2^-(r/s)^2,
                so 3 s leaves 0.2% of a perfectly supported surface outside.
  per region    mean/max deviation of each segmented scan region, so the agent
                knows which face is off ("region 3, the 20 mm bore, +0.8 mm").
  silhouettes   IoU between the CAD rendered from each SfM camera and the SAM 2
                mask - independent of the dense cloud's noise and holes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.geometry import apply_transform, transform_normals

from .. import cad
from ..geometry import raycast
from ..workspace import ScanWorkspace

CAD_SAMPLES = 30000
MIN_SEEING_CAMERAS = 3
MAX_INCIDENCE_COS = 0.5  # 60 degrees from head-on
F_SCORE_FACTOR = 2.5
SPACING_FACTOR = 3.0
SILHOUETTE_VIEWS = 12
SILHOUETTE_STRIDE = 4


@dataclass
class CadMesh:
    vertices: np.ndarray
    triangles: np.ndarray

    @classmethod
    def from_shape(cls, shape, tolerance: float = 0.05) -> "CadMesh":
        return cls(*cad.tessellate(shape, tolerance))

    def scene(self) -> raycast.Scene:
        s = raycast.Scene()
        s.add(self.vertices, self.triangles)
        return s

    def sample(self, count: int = CAD_SAMPLES, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        import trimesh

        m = trimesh.Trimesh(self.vertices, self.triangles, process=False)
        pts, fi = trimesh.sample.sample_surface_even(m, count, seed=seed)
        return pts, m.face_normals[fi]


def _stats(d: np.ndarray, tolerance: float) -> dict:
    if len(d) == 0:
        return {"count": 0}
    return {"count": int(len(d)), "mean_mm": round(float(d.mean()), 4),
            "rms_mm": round(float(np.sqrt(np.mean(d ** 2))), 4),
            "p95_mm": round(float(np.percentile(d, 95)), 4), "max_mm": round(float(d.max()), 4),
            "within_tol_share": round(float(np.mean(d <= tolerance)), 4)}


def observable(points: np.ndarray, normals: np.ndarray, scene: raycast.Scene,
               camera_centers: np.ndarray | None, floor_z: float | None = 0.75) -> np.ndarray:
    """CAD surface samples that at least MIN_SEEING_CAMERAS cameras see unoccluded."""
    ok = np.ones(len(points), dtype=bool)
    if floor_z is not None:
        ok &= points[:, 2] > floor_z
    if camera_centers is None or len(camera_centers) == 0:
        return ok
    seen = np.zeros(len(points), dtype=np.int32)
    for c in camera_centers:
        to_cam = c - points
        dist = np.linalg.norm(to_cam, axis=1)
        dirs = to_cam / dist[:, None]
        facing = np.sum(dirs * normals, axis=1) > MAX_INCIDENCE_COS
        origin = points + dirs * 0.05  # step off the surface before casting
        rays = np.concatenate([origin, dirs], axis=1)
        hit = scene.cast(rays)
        free = hit["t"] >= dist - 0.1
        seen += facing & free
    return ok & (seen >= MIN_SEEING_CAMERAS)


def deviation(mesh: CadMesh, scan_pts: np.ndarray, tolerance: float,
              camera_centers: np.ndarray | None = None, labels: np.ndarray | None = None,
              floor_z: float | None = 0.75) -> dict:
    from scipy.spatial import cKDTree

    scene = mesh.scene()
    d_scan = scene.distance(scan_pts)
    samples, snormals = mesh.sample()
    vis = observable(samples, snormals, scene, camera_centers, floor_z)
    d_cad, _ = cKDTree(scan_pts).query(samples[vis])
    out = {
        "tolerance_mm": tolerance,
        "scan_to_cad": _stats(d_scan, tolerance),
        "cad_to_scan": _stats(d_cad, tolerance),
        "observable_cad_share": round(float(vis.mean()), 4),
        "worst_scan_areas": _worst_clusters(scan_pts, d_scan, tolerance),
        "unsupported_cad_areas": _worst_clusters(samples[vis], d_cad, tolerance),
    }
    if labels is not None:
        out["regions"] = {
            int(r): {"mean_mm": round(float(d_scan[labels == r].mean()), 4),
                     "max_mm": round(float(d_scan[labels == r].max()), 4)}
            for r in np.unique(labels) if r >= 0
        }
    s2c, c2s = out["scan_to_cad"], out["cad_to_scan"]
    sub = scan_pts[:: max(1, len(scan_pts) // 20000)]
    spacing = float(np.median(cKDTree(sub).query(sub, k=2)[0][:, 1])) * np.sqrt(len(sub) / len(scan_pts))
    f_tol = max(F_SCORE_FACTOR * tolerance, SPACING_FACTOR * spacing)
    out["summary"] = {
        "chamfer_mean_mm": round((s2c["mean_mm"] + c2s.get("mean_mm", 0.0)) / 2, 4),
        "f_score": _f_score(np.mean(d_scan <= f_tol), np.mean(d_cad <= f_tol) if len(d_cad) else 0.0),
        "f_score_tol_mm": round(f_tol, 4),
    }
    return out


def _f_score(precision: float, recall: float) -> float:
    return round(float(2 * precision * recall / (precision + recall)), 4) if precision + recall else 0.0


def _worst_clusters(points: np.ndarray, d: np.ndarray, tolerance: float, top: int = 5) -> list[dict]:
    """Connected blobs of out-of-tolerance points, biggest first: where to look."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    bad = np.flatnonzero(d > tolerance)
    if len(bad) == 0:
        return []
    if len(bad) > 20000:
        bad = bad[np.argsort(d[bad])[-20000:]]
    pts = points[bad]
    spacing = np.median(cKDTree(points).query(points[:2000], k=2)[0][:, 1])
    pairs = cKDTree(pts).query_pairs(4 * spacing, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(pts),) * 2)
    _, lab = connected_components(g, directed=False)
    out = []
    for k in np.argsort(-np.bincount(lab))[:top]:
        m = lab == k
        if m.sum() < 5:
            break
        out.append({"points": int(m.sum()), "centroid": [round(float(v), 2) for v in pts[m].mean(0)],
                    "mean_dev_mm": round(float(d[bad][m].mean()), 3),
                    "max_dev_mm": round(float(d[bad][m].max()), 3)})
    return out


def silhouettes(ws: ScanWorkspace, mesh: CadMesh, views: int = SILHOUETTE_VIEWS,
                stride: int = SILHOUETTE_STRIDE) -> dict:
    """IoU of the CAD silhouette vs the SAM 2 mask from evenly spread SfM cameras."""
    import cv2
    import pycolmap

    T = ws.world_to_mm()
    rec = pycolmap.Reconstruction(ws.sfm.sparse_model_dir)
    images = sorted((im for im in rec.images.values() if im.has_pose and ws.mask_path(im.name).exists()),
                    key=lambda im: im.name)
    if not images:
        return {"views": 0}
    pick = [images[i] for i in np.linspace(0, len(images) - 1, min(views, len(images))).astype(int)]
    scene = mesh.scene()
    ious = {}
    for im in pick:
        rays = raycast.colmap_rays(im.camera, im.cam_from_world(), stride)
        rays_mm = np.concatenate([apply_transform(T, rays[..., :3].reshape(-1, 3)),
                                  transform_normals(T, rays[..., 3:].reshape(-1, 3))], axis=1)
        sil = (scene.cast(rays_mm)["geometry"] >= 0).reshape(rays.shape[:2])
        mask = cv2.imread(str(ws.mask_path(im.name)), cv2.IMREAD_GRAYSCALE) > 127
        mask = mask[stride // 2::stride, stride // 2::stride][:sil.shape[0], :sil.shape[1]]
        union = np.logical_or(sil, mask).sum()
        ious[im.name] = round(float(np.logical_and(sil, mask).sum() / union), 4) if union else 1.0
    vals = np.array(list(ious.values()))
    worst = min(ious, key=ious.get)
    return {"views": len(ious), "mean_iou": round(float(vals.mean()), 4),
            "min_iou": round(float(vals.min()), 4), "worst_view": worst, "per_view": ious}


def camera_centers_mm(ws: ScanWorkspace) -> np.ndarray | None:
    import pycolmap

    T = ws.world_to_mm()
    if T is None or not ws.sfm.has_sparse():
        return None
    rec = pycolmap.Reconstruction(ws.sfm.sparse_model_dir)
    centers = np.array([im.projection_center() for im in rec.images.values() if im.has_pose])
    # A few spread-out cameras are enough for the visibility test and keep it fast.
    if len(centers) > 24:
        centers = centers[np.linspace(0, len(centers) - 1, 24).astype(int)]
    return apply_transform(T, centers)
