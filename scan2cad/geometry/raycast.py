"""Headless rendering by CPU ray casting (Open3D RaycastingScene).

No GPU or display needed, so the same code renders the synthetic test videos,
the CAD silhouettes compared against SAM 2 masks, and the preview images the
agent looks at. Camera convention is COLMAP's: x_cam = R @ X + t, +z forward,
pixel centers at (i + 0.5, j + 0.5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PinholeCamera:
    """Undistorted pinhole camera with a world->camera pose."""

    K: np.ndarray  # 3x3
    R: np.ndarray  # 3x3, world -> camera
    t: np.ndarray  # 3
    width: int
    height: int

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    def rays(self, stride: int = 1) -> np.ndarray:
        """(h, w, 6) world-space rays (origin, unit direction) through pixel centers."""
        u = np.arange(0, self.width, stride) + 0.5 * stride
        v = np.arange(0, self.height, stride) + 0.5 * stride
        uu, vv = np.meshgrid(u, v)
        pix = np.stack([uu, vv, np.ones_like(uu)], axis=-1)
        dirs_cam = pix @ np.linalg.inv(self.K).T
        return _world_rays(self, dirs_cam)

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixel coordinates (n, 2) and camera depth (n,) of world points."""
        cam = np.asarray(points) @ self.R.T + self.t
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (cam @ self.K.T)[:, :2] / z[:, None]
        return uv, z


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """World->camera (R, t) for a camera at `eye` looking at `target` (image y points down)."""
    eye, target, up = (np.asarray(x, dtype=float) for x in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-9:  # looking straight along `up`
        x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ eye


def intrinsics(f: float, width: int, height: int) -> np.ndarray:
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=float)


def colmap_rays(camera, cam_from_world, stride: int = 1) -> np.ndarray:
    """(h, w, 6) world rays for a pycolmap camera (distortion included) and pose."""
    u = np.arange(0, camera.width, stride) + 0.5 * stride
    v = np.arange(0, camera.height, stride) + 0.5 * stride
    uu, vv = np.meshgrid(u, v)
    xy = camera.cam_from_img(np.stack([uu.ravel(), vv.ravel()], axis=1))
    dirs_cam = np.concatenate([xy, np.ones((len(xy), 1))], axis=1).reshape(*uu.shape, 3)
    M = cam_from_world.matrix()
    cam = PinholeCamera(np.eye(3), M[:, :3], M[:, 3], camera.width, camera.height)
    return _world_rays(cam, dirs_cam)


def _world_rays(cam: PinholeCamera, dirs_cam: np.ndarray) -> np.ndarray:
    dirs = dirs_cam @ cam.R  # R^T applied to row vectors
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    origin = np.broadcast_to(cam.center, dirs.shape)
    return np.concatenate([origin, dirs], axis=-1).astype(np.float32)


class Scene:
    """Triangle meshes in one ray-casting scene; geometry ids follow insertion order."""

    def __init__(self) -> None:
        import open3d as o3d

        self._scene = o3d.t.geometry.RaycastingScene()

    def add(self, vertices: np.ndarray, triangles: np.ndarray) -> int:
        import open3d as o3d

        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex.positions = o3d.core.Tensor(np.asarray(vertices, dtype=np.float32))
        mesh.triangle.indices = o3d.core.Tensor(np.asarray(triangles, dtype=np.uint32))
        return int(self._scene.add_triangles(mesh))

    def cast(self, rays: np.ndarray) -> dict[str, np.ndarray]:
        """Hit distance `t` (inf on miss), `geometry` id (-1 on miss), unit `normal`."""
        import open3d as o3d

        out = self._scene.cast_rays(o3d.core.Tensor(np.ascontiguousarray(rays, dtype=np.float32)))
        t = out["t_hit"].numpy()
        geom = out["geometry_ids"].numpy().astype(np.int64)
        geom[~np.isfinite(t)] = -1
        return {"t": t, "geometry": geom, "normal": out["primitive_normals"].numpy()}

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Unsigned distance from each point to the nearest surface."""
        import open3d as o3d

        q = o3d.core.Tensor(np.asarray(points, dtype=np.float32))
        return self._scene.compute_distance(q).numpy().astype(float)


def shade(normals: np.ndarray, rays: np.ndarray, hit: np.ndarray,
          albedo: np.ndarray | float = 0.8) -> np.ndarray:
    """Headlight Lambert shading, 0-1, two-sided (normals may face either way)."""
    lam = np.abs(np.sum(normals * rays[..., 3:], axis=-1))
    out = (0.25 + 0.75 * lam) * hit
    return out[..., None] * albedo if np.ndim(albedo) else out * albedo
