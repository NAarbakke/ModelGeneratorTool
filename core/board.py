"""ChArUco board: printable target, detection, and COLMAP-world -> mm alignment.

Primary method: triangulate every detected board corner from the SfM camera
poses, then fit a similarity transform (Umeyama) onto the board's known corner
layout. This uses all corners and the SfM camera model directly. Fallback when
too few corners triangulate: per-frame board pose (solvePnP) and a similarity
fit between the two sets of camera centers.

Output frame: origin at the board center, X along the board's long side, Z up
out of the board, mm.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .geometry import robust_umeyama, scale_of, triangulate

# OpenCV pixel centers are at integer coordinates, COLMAP's at +0.5.
OPENCV_TO_COLMAP_PX = 0.5
MIN_CORNERS_PER_VIEW = 6
MIN_TRIANGULATED = 8


@dataclass(frozen=True)
class BoardSpec:
    squares_x: int = 9
    squares_y: int = 7
    square_mm: float = 25.0
    marker_mm: float = 18.0
    dictionary: str = "DICT_5X5_100"

    @property
    def size_mm(self) -> tuple[float, float]:
        return self.squares_x * self.square_mm, self.squares_y * self.square_mm

    def to_dict(self) -> dict:
        return asdict(self)


def make_board(spec: BoardSpec):
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
    return cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y), spec.square_mm, spec.marker_mm, dictionary
    )


def board_image(spec: BoardSpec, px_per_mm: float, margin_mm: float = 0.0) -> np.ndarray:
    """Grayscale board image; pixel (col, row) = board (x, y) * px_per_mm + margin."""
    w_mm, h_mm = spec.size_mm
    margin = int(round(margin_mm * px_per_mm))
    size = (int(round(w_mm * px_per_mm)) + 2 * margin, int(round(h_mm * px_per_mm)) + 2 * margin)
    return make_board(spec).generateImage(size, marginSize=margin, borderBits=1)


def board_corners_mm(spec: BoardSpec) -> np.ndarray:
    """(n, 3) inner corners in OpenCV board coordinates (x right, y down, z into board)."""
    return np.asarray(make_board(spec).getChessboardCorners(), dtype=float)


def to_output_frame(spec: BoardSpec, board_pts: np.ndarray) -> np.ndarray:
    """OpenCV board coordinates -> output frame: flip y and z (a proper rotation),
    then move the origin to the board center."""
    w, h = spec.size_mm
    return np.asarray(board_pts, dtype=float) * [1.0, -1.0, -1.0] + [-w / 2, h / 2, 0.0]


def detect(gray: np.ndarray, spec: BoardSpec) -> tuple[np.ndarray, np.ndarray]:
    """Corner pixels (COLMAP convention) and ids; empty arrays if not found."""
    import cv2

    detector = cv2.aruco.CharucoDetector(make_board(spec))
    corners, ids, _, _ = detector.detectBoard(gray)
    if corners is None or ids is None or len(ids) < MIN_CORNERS_PER_VIEW:
        return np.empty((0, 2)), np.empty(0, dtype=int)
    return corners.reshape(-1, 2).astype(float) + OPENCV_TO_COLMAP_PX, ids.ravel().astype(int)


# --- camera models -------------------------------------------------------------

def opencv_distortion(camera) -> np.ndarray:
    """OpenCV distortion vector for a pycolmap camera (zeros for pinhole models)."""
    name, p = camera.model.name, np.asarray(camera.params, dtype=float)
    if name == "SIMPLE_RADIAL":
        return np.array([p[3], 0, 0, 0])
    if name == "RADIAL":
        return np.array([p[3], p[4], 0, 0])
    if name == "OPENCV":
        return p[4:8]
    if name in ("SIMPLE_PINHOLE", "PINHOLE"):
        return np.zeros(4)
    raise ValueError(f"camera model {name} not supported for PnP; use the triangulation method")


# --- alignment from a reconstruction ---------------------------------------------

def align(reconstruction, images_dir, spec: BoardSpec) -> dict:
    """Detect the board in every registered image and compute COLMAP world -> mm.

    Returns {"world_to_mm": 4x4, "method", "rms_mm", "views", "corners"}.
    """
    import cv2

    obj_out = to_output_frame(spec, board_corners_mm(spec))
    detections = {}  # image_id -> (pixels, ids)
    for image_id, image in reconstruction.images.items():
        if not image.has_pose:
            continue
        gray = cv2.imread(str(Path(images_dir) / image.name), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        px, ids = detect(gray, spec)
        if len(ids):
            detections[image_id] = (px, ids)
    if len(detections) < 2:
        raise RuntimeError(
            f"ChArUco board found in only {len(detections)} registered frame(s) - keep the "
            "whole board flat, in view and in focus, or calibrate with a known distance instead"
        )

    world, target = _triangulated_corners(reconstruction, detections, obj_out)
    if len(world) >= MIN_TRIANGULATED:
        T, keep = robust_umeyama(world, target)
        method = "triangulated-corners"
        res = np.linalg.norm(world[keep] @ T[:3, :3].T + T[:3, 3] - target[keep], axis=1)
        used = int(keep.sum())
    else:
        T, res, used = _pnp_centers(reconstruction, detections, spec, obj_out)
        method = "pnp-camera-centers"
    return {
        "world_to_mm": T, "method": method, "rms_mm": float(np.sqrt(np.mean(res ** 2))),
        "views": len(detections), "corners": used,
        "scale_mm_per_unit": scale_of(T),
    }


def _triangulated_corners(rec, detections, obj_out):
    obs: dict[int, list] = {}
    for image_id, (px, ids) in detections.items():
        image = rec.images[image_id]
        P = image.cam_from_world().matrix()
        norm = image.camera.cam_from_img(px)
        for cid, xy in zip(ids, norm):
            obs.setdefault(int(cid), []).append((P, xy))
    world, target = [], []
    for cid, views in obs.items():
        if len(views) < 2:
            continue
        X = triangulate([v[0] for v in views], [v[1] for v in views])
        if np.all(np.isfinite(X)):
            world.append(X)
            target.append(obj_out[cid])
    return np.asarray(world).reshape(-1, 3), np.asarray(target).reshape(-1, 3)


def _pnp_centers(rec, detections, spec, obj_out):
    import cv2

    obj_board = board_corners_mm(spec)
    centers_world, centers_mm = [], []
    for image_id, (px, ids) in detections.items():
        image = rec.images[image_id]
        cam = image.camera
        K = cam.calibration_matrix()
        ok, rvec, tvec = cv2.solvePnP(
            obj_board[ids], px - OPENCV_TO_COLMAP_PX, K, opencv_distortion(cam),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            continue
        R_b, _ = cv2.Rodrigues(rvec)
        center_board = -R_b.T @ tvec.ravel()
        centers_mm.append(to_output_frame(spec, center_board[None])[0])
        centers_world.append(image.projection_center())
    if len(centers_world) < 3:
        raise RuntimeError("too few board poses to align (need 3 frames with the board)")
    src, dst = np.asarray(centers_world), np.asarray(centers_mm)
    T, keep = robust_umeyama(src, dst)
    res = np.linalg.norm(src[keep] @ T[:3, :3].T + T[:3, 3] - dst[keep], axis=1)
    return T, res, int(keep.sum())


def calibrate(ws, spec: BoardSpec) -> dict:
    """Align the workspace's SfM model to the board and save it as its calibration.

    ws: a core.workspace.Workspace with a sparse model. Returns the align() result.
    """
    import pycolmap

    from . import scale

    if not ws.has_sparse():
        raise RuntimeError(f"no SfM model in {ws.sparse_model_dir} - reconstruct first")
    rec = pycolmap.Reconstruction(ws.sparse_model_dir)
    res = align(rec, ws.photos_dir, spec)
    scale.save_calibration(ws.scale_path, res["world_to_mm"], f"charuco/{res['method']}",
                           rms_mm=res["rms_mm"], views=res["views"], corners=res["corners"],
                           board=spec.to_dict())
    return res
