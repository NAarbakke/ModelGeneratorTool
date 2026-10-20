"""Split the object cloud into face regions and fit a primitive to each.

Greedy multi-model RANSAC in the style of Schnabel et al. 2007 ("Efficient
RANSAC for point-cloud shape detection"):
  1. Each round proposes plane / sphere / cylinder / cone candidates from
     minimal samples of points + normals, drawn locally around a random seed.
  2. A point supports a candidate if it is within tolerance of the surface AND
     its normal agrees with the surface normal there. The candidate with the
     largest connected support wins, is refined by least squares
     (`primitives.fit_best`, so the preferred adequate kind is kept), and its
     points are removed. Taking the largest candidate first is what stops a
     narrow (locally flat) strip of a cylinder from being claimed as a plane.
  3. Leftover points are grown into smooth regions and fitted (this is where
     tori, which random minimal samples rarely hit, get picked up).
  4. Adjacent regions merge whenever a single model fits their union.

The median rms of the planar regions is reported as the scan's noise floor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import primitives as prim

NORMAL_AGREE_COS = np.cos(np.radians(12))
GROW_ANGLE_COS = np.cos(np.radians(10))
CANDIDATES_PER_ROUND = 60  # per kind
MAX_ROUNDS = 60
MIN_SAMPLES = {"plane": 3, "sphere": 2, "cylinder": 2, "cone": 3}


@dataclass
class Region:
    indices: np.ndarray
    fit: prim.Fit | None
    source: str  # "ransac", "grown" or "merged"

    @property
    def kind(self) -> str:
        return self.fit.kind if self.fit else "freeform"


def point_spacing(points: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    d, _ = cKDTree(points).query(points, k=2)
    return float(np.median(d[:, 1]))


def _components(points: np.ndarray, idx: np.ndarray, radius: float) -> list[np.ndarray]:
    """Connected components of points[idx] under a fixed neighbor radius, largest first."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    if len(idx) == 0:
        return []
    pairs = cKDTree(points[idx]).query_pairs(radius, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(idx),) * 2)
    _, labels = connected_components(g, directed=False)
    return sorted((idx[labels == k] for k in np.unique(labels)), key=len, reverse=True)


# --- model surface normals (for the normal-agreement test) -----------------------

def model_normals(kind: str, p: dict, x: np.ndarray) -> np.ndarray:
    if kind == "plane":
        return np.broadcast_to(p["normal"], x.shape)
    if kind == "sphere":
        d = x - p["center"]
    else:
        a = p["axis_dir"]
        origin = p.get("axis_point", p.get("apex", p.get("center")))
        w = x - origin
        radial = w - np.outer(w @ a, a)
        radial /= np.linalg.norm(radial, axis=1, keepdims=True) + 1e-12
        if kind == "cylinder":
            d = radial
        elif kind == "cone":
            al = np.radians(p["half_angle_deg"])
            d = np.cos(al) * radial - np.sin(al) * a
        else:  # torus: from the nearest spine-circle point to x
            d = w - p["major_radius"] * radial
    return d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)


def support(kind: str, p: dict, x: np.ndarray, n: np.ndarray, tolerance: float) -> np.ndarray:
    ok = np.abs(prim.RESIDUALS[kind](p, x)) <= tolerance
    if ok.any():
        ok[ok] = np.abs(np.sum(model_normals(kind, p, x[ok]) * n[ok], axis=1)) >= NORMAL_AGREE_COS
    return ok


# --- minimal-sample candidates ----------------------------------------------------

def _closest_between_lines(p1, d1, p2, d2):
    """Midpoint of the shortest segment between two lines (None if parallel)."""
    w = p1 - p2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    den = a * c - b * b
    if abs(den) < 1e-9:
        return None
    s = (b * (d2 @ w) - c * (d1 @ w)) / den
    t = (a * (d2 @ w) - b * (d1 @ w)) / den
    return 0.5 * ((p1 + s * d1) + (p2 + t * d2))


def candidate(kind: str, xs: np.ndarray, ns: np.ndarray) -> dict | None:
    if kind == "plane":
        n = np.cross(xs[1] - xs[0], xs[2] - xs[0])
        if np.linalg.norm(n) < 1e-9:
            return None
        return {"normal": prim._unit(n), "point": xs[0]}
    if kind == "sphere":
        c = _closest_between_lines(xs[0], ns[0], xs[1], ns[1])
        if c is None:
            return None
        return {"center": c, "radius": float(np.mean(np.linalg.norm(xs[:2] - c, axis=1)))}
    if kind == "cylinder":
        a = np.cross(ns[0], ns[1])
        if np.linalg.norm(a) < 1e-3:
            return None
        a = prim._unit(a)
        # project both normal lines onto the plane _|_ a and intersect them
        q = [x - (x @ a) * a for x in xs[:2]]
        m = [n - (n @ a) * a for n in ns[:2]]
        c = _closest_between_lines(q[0], m[0], q[1], m[1])
        if c is None:
            return None
        r = float(np.mean([np.linalg.norm(q[i] - c) for i in range(2)]))
        return {"axis_point": c, "axis_dir": a, "radius": r}
    if kind == "cone":
        try:  # the apex lies on all three tangent planes
            apex = np.linalg.solve(ns[:3], np.sum(ns[:3] * xs[:3], axis=1))
        except np.linalg.LinAlgError:
            return None
        dirs = xs[:3] - apex
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        a = np.cross(dirs[1] - dirs[0], dirs[2] - dirs[0])
        if np.linalg.norm(a) < 1e-9:
            return None
        a = prim._unit(a)
        if (dirs @ a).mean() < 0:
            a = -a
        half = float(np.degrees(np.arccos(np.clip((dirs @ a).mean(), -1, 1))))
        if not 2.0 < half < 88.0:
            return None
        return {"apex": apex, "axis_dir": a, "half_angle_deg": half}
    raise ValueError(kind)


def ransac_regions(points: np.ndarray, normals: np.ndarray, tolerance: float, min_size: int,
                   spacing: float, rng: np.random.Generator) -> tuple[list[Region], np.ndarray]:
    from scipy.spatial import cKDTree

    span = float(np.max(np.ptp(points, axis=0)))
    remaining = np.arange(len(points))
    regions: list[Region] = []
    for _ in range(MAX_ROUNDS):
        if len(remaining) < min_size:
            break
        x, n = points[remaining], normals[remaining]
        tree = cKDTree(x)
        local_r = max(0.15 * span, 10 * spacing)
        best = None  # (support count, kind, params, mask)
        for kind, k in MIN_SAMPLES.items():
            for _ in range(CANDIDATES_PER_ROUND):
                seed = int(rng.integers(len(x)))
                near = tree.query_ball_point(x[seed], local_r)
                if len(near) < k:
                    continue
                pick = [seed, *rng.choice(near, k - 1, replace=False)]
                c = candidate(kind, x[pick], n[pick])
                if c is None or c.get("radius", 0) > 20 * span:
                    continue
                mask = support(kind, c, x, n, tolerance)
                cnt = int(mask.sum())
                if best is None or cnt > best[0]:
                    best = (cnt, kind, c, mask)
        if best is None or best[0] < min_size:
            break
        comp = _components(points, remaining[best[3]], 3 * spacing)[0]
        if len(comp) < min_size:
            break
        f, _ = prim.fit_best(points[comp], normals[comp], tolerance)
        if f is None:
            break
        # Re-collect support with the refined model: it is usually larger.
        refined = _components(points, remaining[support(f.kind, f.params, x, n, tolerance)],
                              3 * spacing)
        if refined and len(refined[0]) > len(comp):
            f2, _ = prim.fit_best(points[refined[0]], normals[refined[0]], tolerance)
            if f2 is not None:
                comp, f = refined[0], f2
        regions.append(Region(comp, f, "ransac"))
        remaining = np.setdiff1d(remaining, comp)
    return regions, remaining


def grow_regions(points: np.ndarray, normals: np.ndarray, idx: np.ndarray, spacing: float,
                 min_size: int) -> list[np.ndarray]:
    """Smoothness-constrained region growing over points[idx]."""
    from scipy.spatial import cKDTree

    if len(idx) == 0:
        return []
    pts, nrm = points[idx], normals[idx]
    neigh = cKDTree(pts).query_ball_point(pts, 2.5 * spacing)
    label = np.full(len(idx), -1)
    out = []
    for seed in range(len(idx)):
        if label[seed] >= 0:
            continue
        label[seed] = len(out)
        stack, members = [seed], [seed]
        while stack:
            i = stack.pop()
            for j in neigh[i]:
                if label[j] < 0 and abs(nrm[i] @ nrm[j]) >= GROW_ANGLE_COS:
                    label[j] = label[seed]
                    stack.append(j)
                    members.append(j)
        out.append(np.asarray(members))
    return [idx[m] for m in out if len(m) >= min_size]


def merge_regions(points: np.ndarray, normals: np.ndarray, regions: list[Region],
                  tolerance: float, spacing: float) -> list[Region]:
    """Repeatedly merge an adjacent pair whose union one model explains."""
    tried: set[tuple[int, int]] = set()  # pairs (by first point index) already rejected
    while True:
        adj = adjacency(points, regions, 3 * spacing)
        for i, r in enumerate(regions):
            for j in adj[i]:
                key = tuple(sorted((int(r.indices[0]), int(regions[j].indices[0]))))
                if j <= i or key in tried:
                    continue
                union = np.concatenate([r.indices, regions[j].indices])
                f, _ = prim.fit_best(points[union], normals[union], tolerance)
                if f is not None:
                    regions = [q for k, q in enumerate(regions) if k not in (i, j)]
                    regions.append(Region(union, f, "merged"))
                    break
                tried.add(key)
            else:
                continue
            break
        else:
            return regions


def segment(points: np.ndarray, normals: np.ndarray, tolerance: float | None = None,
            min_size: int | None = None, seed: int = 0) -> tuple[list[Region], dict]:
    """tolerance: inlier band in mm; default 3x the estimated scan noise sigma."""
    rng = np.random.default_rng(seed)
    spacing = point_spacing(points)
    noise = prim.estimate_noise(points)
    if tolerance is None:
        tolerance = max(3 * noise, 0.05)
    min_size = min_size or max(30, len(points) // 1000)
    regions, rest = ransac_regions(points, normals, tolerance, min_size, spacing, rng)
    for grown in grow_regions(points, normals, rest, spacing, min_size):
        f, _ = prim.fit_best(points[grown], normals[grown], tolerance)
        regions.append(Region(grown, f, "grown"))
    regions = merge_regions(points, normals, regions, tolerance, spacing)
    regions.sort(key=lambda r: -len(r.indices))
    assigned = sum(len(r.indices) for r in regions)
    plane_rms = [r.fit.rms for r in regions if r.kind == "plane"]
    stats = {
        "points": len(points), "spacing_mm": round(spacing, 4),
        "tolerance_mm": round(tolerance, 4), "noise_sigma_estimate_mm": round(noise, 4),
        "regions": len(regions), "unassigned_share": round(1 - assigned / len(points), 3),
        "kinds": {k: sum(r.kind == k for r in regions) for k in (*prim.KINDS, "freeform")},
        "noise_floor_mm": round(float(np.median(plane_rms)), 4) if plane_rms else None,
    }
    return regions, stats


def adjacency(points: np.ndarray, regions: list[Region], radius: float) -> list[list[int]]:
    """Region ids whose points come within `radius` of each other."""
    from scipy.spatial import cKDTree

    label = np.full(len(points), -1)
    for i, r in enumerate(regions):
        label[r.indices] = i
    has = label >= 0
    lab = label[has]
    adj = [set() for _ in regions]
    for i, j in cKDTree(points[has]).query_pairs(radius, output_type="ndarray"):
        a, b = lab[i], lab[j]
        if a != b:
            adj[a].add(int(b))
            adj[b].add(int(a))
    return [sorted(s) for s in adj]


def write_regions(regions: list[Region], stats: dict, points: np.ndarray,
                  json_path: Path, ply_path: Path) -> list[dict]:
    import open3d as o3d

    adj = adjacency(points, regions, 3 * stats["spacing_mm"])
    area_per_point = stats["spacing_mm"] ** 2
    out = []
    for i, r in enumerate(regions):
        p = points[r.indices]
        entry = {"id": i, "kind": r.kind, "points": int(len(r.indices)), "source": r.source,
                 "approx_area_mm2": round(len(r.indices) * area_per_point, 1),
                 "centroid": [round(float(v), 3) for v in p.mean(0)],
                 "bbox_min": [round(float(v), 3) for v in p.min(0)],
                 "bbox_max": [round(float(v), 3) for v in p.max(0)],
                 "neighbors": adj[i]}
        if r.fit is not None:
            entry["fit"] = r.fit.summary()
        out.append(entry)
    json_path.write_text(json.dumps({"stats": stats, "regions": out}, indent=1))

    rng = np.random.default_rng(0)
    colors = np.full((len(points), 3), 0.3)
    labels = np.full(len(points), -1)
    for i, r in enumerate(regions):
        colors[r.indices] = rng.uniform(0.2, 1.0, 3)
        labels[r.indices] = i
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(ply_path), pcd)
    np.save(ply_path.with_suffix(".labels.npy"), labels)
    return out
