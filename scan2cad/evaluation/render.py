"""Preview images: shaded CAD, scan points colored by deviation, from named views.

Rendering is CPU ray casting (see raycast.py), so it works headless - which is
how the MCP server shows the agent what its model looks like against the scan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..geometry import raycast

VIEWS = {  # azimuth, elevation in degrees; board frame, Z up
    "iso": (-45, 30), "iso_back": (135, 30), "top": (0, 89.9), "bottom": (0, -89.9),
    "front": (-90, 0), "back": (90, 0), "right": (0, 0), "left": (180, 0),
}
CAD_COLOR = np.array([0.62, 0.68, 0.78])


def view_camera(view: str, center: np.ndarray, radius: float, width: int, height: int,
                fov_deg: float = 35.0) -> raycast.PinholeCamera:
    if view not in VIEWS:
        raise KeyError(f"unknown view {view!r}; choose from {', '.join(VIEWS)}")
    az, el = np.radians(VIEWS[view])
    dist = radius / np.tan(np.radians(fov_deg) / 2) * 1.15
    eye = center + dist * np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
    up = (0, 1.0, 0) if abs(VIEWS[view][1]) > 80 else (0, 0, 1.0)
    R, t = raycast.look_at(eye, center, up)
    f = (height / 2) / np.tan(np.radians(fov_deg) / 2)
    return raycast.PinholeCamera(raycast.intrinsics(f, width, height), R, t, width, height)


def deviation_colors(d: np.ndarray, tolerance: float) -> np.ndarray:
    """Green within tolerance, through yellow, to red at 3x tolerance."""
    x = np.clip(np.asarray(d) / max(tolerance, 1e-9), 0, 3)
    g = np.where(x <= 1, 1.0, np.clip(1 - (x - 1) / 2, 0, 1))
    r = np.where(x <= 1, x * 0.6, 1.0)
    return np.stack([r, g, np.full_like(r, 0.15)], axis=1)


def render(view: str, mesh=None, scan_pts: np.ndarray | None = None,
           scan_colors: np.ndarray | None = None, size: tuple[int, int] = (800, 600),
           point_px: int = 2) -> np.ndarray:
    """RGB uint8 image. scan_colors: per-point RGB 0-1 (e.g. deviation_colors)."""
    width, height = size
    geo = [g for g in (mesh.vertices if mesh is not None else None, scan_pts) if g is not None]
    allpts = np.concatenate(geo)
    lo, hi = allpts.min(0), allpts.max(0)
    center, radius = (lo + hi) / 2, float(np.linalg.norm(hi - lo)) / 2 + 1e-6
    cam = view_camera(view, center, radius, width, height)

    img = np.ones((height, width, 3)) * np.linspace(0.97, 0.88, height)[:, None, None]
    depth = np.full((height, width), np.inf)
    if mesh is not None:
        scene = raycast.Scene()
        scene.add(mesh.vertices, mesh.triangles)
        rays = cam.rays()
        hit = scene.cast(rays)
        mask = hit["geometry"] >= 0
        shade = raycast.shade(hit["normal"], rays, mask)
        img[mask] = shade[mask][:, None] * CAD_COLOR
        depth = np.where(mask, hit["t"], np.inf)

    if scan_pts is not None and len(scan_pts):
        colors = scan_colors if scan_colors is not None else np.tile([0.2, 0.2, 0.2], (len(scan_pts), 1))
        uv, z = cam.project(scan_pts)
        dist = np.linalg.norm(scan_pts - cam.center, axis=1)
        col, row = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
        ok = (z > 0) & (col >= 0) & (col < width) & (row >= 0) & (row < height)
        # draw points not hidden behind the CAD surface (with a little slack), far to near
        slack = 0.01 * radius
        ok[ok] &= dist[ok] <= depth[row[ok], col[ok]] + slack
        order = np.flatnonzero(ok)[np.argsort(-dist[ok])]
        r = point_px // 2
        for dy in range(-r, point_px - r):
            for dx in range(-r, point_px - r):
                rr = np.clip(row[order] + dy, 0, height - 1)
                cc = np.clip(col[order] + dx, 0, width - 1)
                img[rr, cc] = colors[order]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def montage(images: list[np.ndarray], labels: list[str], cols: int = 2) -> np.ndarray:
    import cv2

    h, w = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    out = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)
    for i, (im, lab) in enumerate(zip(images, labels)):
        r, c = divmod(i, cols)
        tile = im.copy()
        cv2.putText(tile, lab, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
        out[r * h:(r + 1) * h, c * w:(c + 1) * w] = tile
    return out


def save_png(img: np.ndarray, path: Path) -> Path:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img[..., ::-1])
    return path
