"""Rigid/similarity transforms and triangulation shared by every tool. Pure numpy."""

from __future__ import annotations

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """4x4 similarity T minimizing ||dst - (s R src + t)||^2 (Umeyama 1991)."""
    src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    if len(src) < 3:
        raise ValueError("need at least 3 point pairs")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var_s = (xs ** 2).sum() / len(src)
    if var_s < 1e-12:
        raise ValueError("source points are degenerate (all coincide)")
    s = np.trace(np.diag(S) @ D) / var_s
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = mu_d - s * R @ mu_s
    return T


def robust_umeyama(src: np.ndarray, dst: np.ndarray, iters: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Umeyama with iterative rejection of pairs beyond 3x the median residual."""
    keep = np.ones(len(src), dtype=bool)
    for _ in range(iters):
        T = umeyama(src[keep], dst[keep])
        res = np.linalg.norm(src @ T[:3, :3].T + T[:3, 3] - dst, axis=1)
        new_keep = res <= max(3 * np.median(res[keep]), 1e-9)
        if new_keep.sum() < 3 or (new_keep == keep).all():
            break
        keep = new_keep
    return T, keep


def triangulate(projections: list[np.ndarray], points_norm: list[np.ndarray]) -> np.ndarray:
    """Linear (DLT) triangulation from >= 2 views.

    projections: 3x4 [R|t] per view; points_norm: normalized image coords (x, y).
    """
    rows = []
    for P, (x, y) in zip(projections, points_norm):
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.asarray(rows))
    X = Vt[-1]
    return X[:3] / X[3]


def apply_transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 (similarity) transform to (n, 3) points."""
    pts = np.asarray(points, dtype=float)
    return pts @ T[:3, :3].T + T[:3, 3]


def transform_normals(T: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Rotate normals by the rotation part of a similarity transform (scale removed)."""
    R = T[:3, :3] / np.cbrt(np.linalg.det(T[:3, :3]))
    return np.asarray(normals, dtype=float) @ R.T


def scale_of(T: np.ndarray) -> float:
    """Uniform scale factor of a similarity transform."""
    return float(np.cbrt(np.linalg.det(np.asarray(T, dtype=float)[:3, :3])))
