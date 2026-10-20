"""Synthetic captures with exact ground truth: a known part on a ChArUco board,
filmed by an orbiting camera.

The part carries a procedural speckle texture (what scanning spray gives a real
part), so SfM finds features on it. Outputs:
  <out>/capture.mp4      the "phone video"
  <out>/gt/part.step     ground-truth CAD (mm, board frame)
  <out>/gt/masks/*.png   exact object masks per video frame
  <out>/gt/cameras.json  K and world->camera poses per video frame
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core import board

from .. import cad
from ..geometry import raycast
from .parts import PARTS

BOARD_PX_PER_MM = 8.0


class SolidTexture:
    """Smooth 3D noise in [0, 1]: a solid texture, so it can't swim between frames."""

    def __init__(self, cell_mm: float, rng: np.random.Generator, octaves: int = 3) -> None:
        self.cell_mm = cell_mm
        self.grids = [rng.random((48, 48, 48)) for _ in range(octaves)]

    def __call__(self, points: np.ndarray) -> np.ndarray:
        from scipy.ndimage import map_coordinates

        out = np.zeros(len(points))
        for o, grid in enumerate(self.grids):
            coords = (points / (self.cell_mm / 2 ** o)).T
            out += map_coordinates(grid, coords, order=1, mode="grid-wrap") * 0.5 ** o
        return out / sum(0.5 ** o for o in range(len(self.grids)))


def _board_sampler(spec: board.BoardSpec):
    from scipy.ndimage import map_coordinates

    img = board.board_image(spec, BOARD_PX_PER_MM).astype(float) / 255.0
    w, h = spec.size_mm

    def sample(xy: np.ndarray) -> np.ndarray:
        col = (xy[:, 0] + w / 2) * BOARD_PX_PER_MM - 0.5
        row = (h / 2 - xy[:, 1]) * BOARD_PX_PER_MM - 0.5
        return map_coordinates(img, [row, col], order=1, mode="nearest")

    return sample


def orbit_cameras(n: int, target: np.ndarray, radius: float, width: int, height: int,
                  focal: float, rng: np.random.Generator) -> list[raycast.PinholeCamera]:
    K = raycast.intrinsics(focal, width, height)
    cams = []
    for i in range(n):
        az = 2 * np.pi * i / n
        el = np.radians(38 + 14 * np.sin(2 * az))  # rise and fall twice per orbit
        r = radius * (1 + 0.08 * np.sin(3 * az))
        eye = target + r * np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
        R, t = raycast.look_at(eye, target + rng.normal(0, 2.0, 3))
        cams.append(raycast.PinholeCamera(K, R, t, width, height))
    return cams


def render_capture(part: str, out_dir: Path, frames: int = 120, width: int = 960,
                   height: int = 540, focal: float = 820.0, supersample: int = 1,
                   blur_frames: tuple[int, ...] = (7, 41, 77), seed: int = 0,
                   spec: board.BoardSpec = board.BoardSpec(), fps: float = 15.0) -> dict:
    import cv2

    rng = np.random.default_rng(seed)
    gt_dir = out_dir / "gt"
    (gt_dir / "masks").mkdir(parents=True, exist_ok=True)

    shape = _build_part(part, gt_dir)
    verts, tris = cad.tessellate(shape, 0.05)
    scene = raycast.Scene()
    part_id = scene.add(verts, tris)
    bw, bh = spec.size_mm
    board_id = scene.add(*_quad(bw / 2, bh / 2, 0.0))
    floor_id = scene.add(*_quad(3000, 3000, -1.0))

    size = np.ptp(verts, axis=0)
    target = np.array([0.0, 0.0, size[2] / 2])
    radius = 1.25 * max(np.linalg.norm(size), np.hypot(bw, bh) * 0.6)
    cams = orbit_cameras(frames, target, radius, width, height, focal, rng)
    board_texture = _board_sampler(spec)
    textures = {part_id: ([0.55, 0.62, 0.78], SolidTexture(4.0, rng)),
                floor_id: ([0.55, 0.5, 0.42], SolidTexture(12.0, rng))}

    writer = cv2.VideoWriter(str(out_dir / "capture.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    light = np.array([0.4, -0.3, 0.87])
    light /= np.linalg.norm(light)
    for i, cam in enumerate(cams):
        hi = raycast.PinholeCamera(cam.K * [[supersample], [supersample], [1]], cam.R, cam.t,
                                   width * supersample, height * supersample)
        rays = hi.rays()
        hit = scene.cast(rays)
        geom, t = hit["geometry"], np.where(np.isfinite(hit["t"]), hit["t"], 0.0)
        pts = rays[..., :3] + rays[..., 3:] * t[..., None]
        color = np.empty(geom.shape + (3,))
        color[:] = np.linspace([0.62, 0.66, 0.72], [0.85, 0.87, 0.9], geom.shape[0])[:, None, :]

        for gid, (base, texture) in textures.items():
            sel = geom == gid
            n = texture(pts[sel])
            color[sel] = np.asarray(base) * (0.15 + 1.5 * (n - 0.5) + 0.75)[:, None]
        sel = geom == board_id
        color[sel] = board_texture(pts[sel][:, :2])[:, None] * [0.95, 0.95, 0.93]

        hitmask = geom >= 0
        lam = np.abs(np.sum(hit["normal"] * light, axis=-1))
        color[hitmask] *= (0.45 + 0.55 * lam[hitmask])[:, None]

        img = cv2.resize((np.clip(color, 0, 1) * 255).astype(np.uint8)[..., ::-1],
                         (width, height), interpolation=cv2.INTER_AREA)
        if i in blur_frames:
            k = np.zeros((1, 21))
            k[0, :] = 1 / 21
            img = cv2.filter2D(img, -1, k)
        writer.write(img)
        mask = cv2.resize((geom == part_id).astype(np.uint8) * 255, (width, height),
                          interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(gt_dir / "masks" / f"{i:05d}.png"), np.where(mask >= 128, 255, 0).astype(np.uint8))
    writer.release()

    centroid = verts.mean(axis=0)
    uv, _ = cams[0].project(centroid[None])
    gt = {
        "part": part, "frames": frames, "fps": fps, "width": width, "height": height,
        "board": spec.to_dict(), "blur_frames": list(blur_frames),
        "click_frame0_px": [float(uv[0, 0]), float(uv[0, 1])],
        "cameras": [{"K": c.K.tolist(), "R": c.R.tolist(), "t": c.t.tolist()} for c in cams],
    }
    (gt_dir / "cameras.json").write_text(json.dumps(gt, indent=1))
    return gt


def _build_part(part: str, gt_dir: Path):
    if part not in PARTS:
        raise KeyError(f"unknown part {part!r}; choose from {', '.join(PARTS)}")
    res = cad.run_script(PARTS[part], gt_dir, "part")
    if not res.get("ok"):
        raise RuntimeError(res["error"])
    return cad.load_step(gt_dir / "part.step")


def _quad(hx: float, hy: float, z: float) -> tuple[np.ndarray, np.ndarray]:
    v = np.array([[-hx, -hy, z], [hx, -hy, z], [hx, hy, z], [-hx, hy, z]])
    return v, np.array([[0, 1, 2], [0, 2, 3]])
