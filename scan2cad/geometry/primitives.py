"""Least-squares fitting of planes, spheres, cylinders, cones and tori to points + normals.

Each fit starts from a closed-form initialization that uses the normals, then is
refined with scipy.optimize.least_squares (soft-L1 loss, so stray points pull
less). `fit_best` returns the simplest model that explains the region within
tolerance; regions nothing explains stay freeform (spline). Narrow bands are
genuinely ambiguous (a 3 mm chamfer band fits a sphere as well as a cone), so the
tie-break follows how common each feature is in machined parts, and every other
kind that also fits is reported as `also_fits`.

All residuals are signed point-to-surface distances in the input units (mm).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

KINDS = ("plane", "cylinder", "cone", "sphere", "torus")  # tie-break order: most common first
INLIER_SHARE = 0.9  # share of points within the tolerance band for a fit to be accepted
# A fit is accepted when rms <= tolerance * RMS_LIMIT_RATIO. With tolerance = 3 sigma (the
# inlier band) that is ~1.5 sigma: loose enough for honest fits, too tight for a wrong model
# that merely stays inside the band (e.g. one torus swallowing a plane and a fillet).
RMS_LIMIT_RATIO = 0.5


@dataclass
class Fit:
    kind: str
    params: dict
    rms: float
    inlier_share: float
    residuals: np.ndarray = field(repr=False)
    also_fits: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        out = {"kind": self.kind, "rms_mm": round(self.rms, 4),
               "inlier_share": round(self.inlier_share, 3), **_round(self.params)}
        if self.also_fits:
            out["also_fits"] = self.also_fits
        return out


# --- small geometry helpers ------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _dir_from_angles(theta: float, phi: float) -> np.ndarray:
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def _angles_from_dir(d: np.ndarray) -> tuple[float, float]:
    d = _unit(d)
    return float(np.arccos(np.clip(d[2], -1, 1))), float(np.arctan2(d[1], d[0]))


def _basis(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors orthogonal to a and each other."""
    helper = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = _unit(np.cross(a, helper))
    return u, np.cross(a, u)


def _fit_circle_2d(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Algebraic (Kasa) circle fit."""
    A = np.column_stack([2 * xy, np.ones(len(xy))])
    b = (xy ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]
    return c, float(np.sqrt(max(sol[2] + c @ c, 1e-12)))


def _smallest_eigvec(m: np.ndarray, center: bool) -> np.ndarray:
    x = m - m.mean(0) if center else m
    _, vecs = np.linalg.eigh(x.T @ x)
    return vecs[:, 0]


def estimate_noise(points: np.ndarray, k: int = 24) -> float:
    """Scan noise sigma: median rms distance of points to their local PCA plane.

    On curved surfaces the neighborhood is small enough that curvature adds little.
    """
    from scipy.spatial import cKDTree

    sample = points if len(points) <= 20000 else points[
        np.random.default_rng(0).choice(len(points), 20000, replace=False)]
    _, nb = cKDTree(points).query(sample, k=k)
    nbh = points[nb] - points[nb].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nbh, nbh) / k
    lam = np.linalg.eigvalsh(cov)[:, 0]
    return float(np.sqrt(np.median(np.maximum(lam, 0)) * k / (k - 3)))


def axis_snap_deg(direction: np.ndarray) -> tuple[str, float]:
    """Nearest coordinate axis and the angle to it, for axis directions."""
    d = np.abs(_unit(np.asarray(direction)))
    i = int(d.argmax())
    return "XYZ"[i], float(np.degrees(np.arccos(np.clip(d[i], -1, 1))))


def _round(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            out[k] = [round(float(x), 4) for x in v]
        elif isinstance(v, float):
            out[k] = round(v, 4)
        else:
            out[k] = v
    return out


# --- residuals -------------------------------------------------------------------

def plane_residuals(p: dict, x: np.ndarray) -> np.ndarray:
    return (x - p["point"]) @ p["normal"]


def sphere_residuals(p: dict, x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x - p["center"], axis=1) - p["radius"]


def cylinder_residuals(p: dict, x: np.ndarray) -> np.ndarray:
    w = x - p["axis_point"]
    return np.linalg.norm(np.cross(w, p["axis_dir"]), axis=1) - p["radius"]


def cone_residuals(p: dict, x: np.ndarray) -> np.ndarray:
    w = x - p["apex"]
    h = w @ p["axis_dir"]
    rho = np.linalg.norm(w - np.outer(h, p["axis_dir"]), axis=1)
    a = np.radians(p["half_angle_deg"])
    return rho * np.cos(a) - h * np.sin(a)


def torus_residuals(p: dict, x: np.ndarray) -> np.ndarray:
    w = x - p["center"]
    h = w @ p["axis_dir"]
    rho = np.linalg.norm(w - np.outer(h, p["axis_dir"]), axis=1)
    return np.hypot(rho - p["major_radius"], h) - p["minor_radius"]


RESIDUALS = {"plane": plane_residuals, "sphere": sphere_residuals,
             "cylinder": cylinder_residuals, "cone": cone_residuals, "torus": torus_residuals}


# --- fits ------------------------------------------------------------------------

def fit_plane(x: np.ndarray, n: np.ndarray | None = None) -> dict:
    c = x.mean(0)
    normal = _smallest_eigvec(x, center=True)
    if n is not None and (n @ normal).mean() < 0:
        normal = -normal  # agree with the outward point normals
    return {"normal": normal, "point": c}


def fit_sphere(x: np.ndarray, n: np.ndarray | None = None, scale: float = 1.0) -> dict:
    A = np.column_stack([2 * x, np.ones(len(x))])
    sol, *_ = np.linalg.lstsq(A, (x ** 2).sum(1), rcond=None)
    c0 = sol[:3]
    r0 = np.sqrt(max(sol[3] + c0 @ c0, 1e-12))

    def f(q):
        return np.linalg.norm(x - q[:3], axis=1) - q[3]

    q = least_squares(f, np.r_[c0, r0], loss="soft_l1", f_scale=scale).x
    return {"center": q[:3], "radius": abs(float(q[3]))}


def fit_cylinder(x: np.ndarray, n: np.ndarray, scale: float = 1.0) -> dict:
    a0 = _smallest_eigvec(n, center=False)  # cylinder normals are all perpendicular to the axis
    u, v = _basis(a0)
    c2, r0 = _fit_circle_2d(np.column_stack([x @ u, x @ v]))
    p0 = c2[0] * u + c2[1] * v
    th, ph = _angles_from_dir(a0)
    origin = x.mean(0)

    def unpack(q):
        a = _dir_from_angles(q[0], q[1])
        bu, bv = _basis(a)
        p = origin - (origin @ a) * a + q[2] * bu + q[3] * bv  # point on axis, 2 dof
        return a, p, q[4]

    bu0, bv0 = _basis(_dir_from_angles(th, ph))
    p_init = p0 - (origin - (origin @ a0) * a0)  # p0 and the projected origin are both _|_ a0

    def f(q):
        a, p, r = unpack(q)
        return np.linalg.norm(np.cross(x - p, a), axis=1) - r

    q0 = np.r_[th, ph, p_init @ bu0, p_init @ bv0, r0]
    q = least_squares(f, q0, loss="soft_l1", f_scale=scale).x
    a, p, r = unpack(q)
    p = p + ((x.mean(0) - p) @ a) * a  # report the axis point nearest the data centroid
    return {"axis_point": p, "axis_dir": a, "radius": abs(float(r))}


def fit_cone(x: np.ndarray, n: np.ndarray, scale: float = 1.0) -> dict:
    a0 = _smallest_eigvec(n, center=True)  # cone normals make a constant angle with the axis
    s = float(np.clip((n @ a0).mean(), -0.99, 0.99))
    if s > 0:
        a0, s = -a0, -s  # orient the axis from the apex into the cone
    # The apex lies on every tangent plane: n_i . (x_i - apex) = 0.
    apex0, *_ = np.linalg.lstsq(n, (n * x).sum(1), rcond=None)
    alpha0 = np.arcsin(abs(s))
    th, ph = _angles_from_dir(a0)

    def unpack(q):
        return q[:3], _dir_from_angles(q[3], q[4]), q[5]

    def f(q):
        apex, a, alpha = unpack(q)
        w = x - apex
        h = w @ a
        rho = np.linalg.norm(w - np.outer(h, a), axis=1)
        return rho * np.cos(alpha) - h * np.sin(alpha)

    q = least_squares(f, np.r_[apex0, th, ph, alpha0], loss="soft_l1", f_scale=scale).x
    apex, a, alpha = unpack(q)
    if (x - apex).mean(0) @ a < 0:
        a = -a
    return {"apex": apex, "axis_dir": a, "half_angle_deg": float(np.degrees(abs(alpha)))}


def fit_torus(x: np.ndarray, n: np.ndarray, scale: float = 1.0) -> dict:
    """Init: for the right minor radius r, the points x - r*n lie on the spine circle."""
    span = float(np.max(np.ptp(x, axis=0)))
    best = None
    for r in np.r_[-np.geomspace(span, span * 1e-3, 40), np.geomspace(span * 1e-3, span, 40)]:
        s = x - r * n
        c = s.mean(0)
        a = _smallest_eigvec(s, center=True)
        u, v = _basis(a)
        c2, R = _fit_circle_2d(np.column_stack([(s - c) @ u, (s - c) @ v]))
        center = c + c2[0] * u + c2[1] * v
        spine_err = np.abs(np.linalg.norm((s - center) - np.outer((s - center) @ a, a), axis=1) - R)
        err = spine_err.mean() + np.abs((s - center) @ a).mean()
        if best is None or err < best[0]:
            best = (err, center, a, R, abs(r))
    _, c0, a0, R0, r0 = best
    th, ph = _angles_from_dir(a0)

    def unpack(q):
        return q[:3], _dir_from_angles(q[3], q[4]), q[5], q[6]

    def f(q):
        c, a, R, r = unpack(q)
        w = x - c
        h = w @ a
        rho = np.linalg.norm(w - np.outer(h, a), axis=1)
        return np.hypot(rho - R, h) - r

    q = least_squares(f, np.r_[c0, th, ph, R0, r0], loss="soft_l1", f_scale=scale).x
    c, a, R, r = unpack(q)
    return {"center": c, "axis_dir": a, "major_radius": abs(float(R)), "minor_radius": abs(float(r))}


FITTERS = {"plane": fit_plane, "sphere": fit_sphere, "cylinder": fit_cylinder,
           "cone": fit_cone, "torus": fit_torus}


def fit(kind: str, x: np.ndarray, n: np.ndarray, tolerance: float) -> Fit | None:
    """Fit one model kind; None if the optimizer fails or the result is degenerate."""
    try:
        if kind == "plane":
            params = fit_plane(x, n)
        else:
            params = FITTERS[kind](x, n, scale=tolerance)
    except (np.linalg.LinAlgError, ValueError):
        return None
    res = RESIDUALS[kind](params, x)
    if not np.all(np.isfinite(res)):
        return None
    _describe(kind, params, x)
    return Fit(kind, params, float(np.sqrt(np.mean(res ** 2))),
               float(np.mean(np.abs(res) <= tolerance)), res)


def fit_best(x: np.ndarray, n: np.ndarray, tolerance: float,
             kinds: tuple[str, ...] = KINDS) -> tuple[Fit | None, list[Fit]]:
    """First kind (in KINDS order) that fits: rms <= tolerance * RMS_LIMIT_RATIO and
    >= INLIER_SHARE of points inside the tolerance band. Also returns all attempted fits."""
    fits = [f for k in kinds if (f := fit(k, x, n, tolerance)) is not None]
    limit = tolerance * RMS_LIMIT_RATIO
    ok = [f for f in fits
          if f.rms <= limit and f.inlier_share >= INLIER_SHARE and _plausible(f, x)]
    if not ok:
        return None, fits
    ok[0].also_fits = [f.kind for f in ok[1:]]
    return ok[0], fits


def _plausible(f: Fit, x: np.ndarray) -> bool:
    """Reject numerically fine but meaningless fits (e.g. a 'sphere' of radius 1e5 mm)."""
    span = float(np.max(np.ptp(x, axis=0)))
    p = f.params
    if f.kind in ("sphere", "cylinder"):
        return p["radius"] < 20 * span
    if f.kind == "cone":
        return 1.0 < p["half_angle_deg"] < 89.0
    if f.kind == "torus":  # minor >= major is a self-intersecting spindle torus: never a fillet
        return p["minor_radius"] < p["major_radius"] < 50 * span
    return True


def _describe(kind: str, p: dict, x: np.ndarray) -> None:
    """Add human/agent-friendly derived values: extents, coverage, snapping hints."""
    if kind == "plane":
        p["offset"] = float(p["point"] @ p["normal"])
        p["nearest_axis"], p["axis_angle_deg"] = axis_snap_deg(p["normal"])
        return
    if "axis_dir" in p:
        a = p["axis_dir"]
        p["nearest_axis"], p["axis_angle_deg"] = axis_snap_deg(a)
        origin = p.get("axis_point", p.get("center", p.get("apex")))
        h = (x - origin) @ a
        p["axial_range"] = np.array([h.min(), h.max()])
        u, v = _basis(a)
        ang = np.sort(np.arctan2((x - origin) @ v, (x - origin) @ u))
        gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
        p["arc_coverage_deg"] = float(np.degrees(2 * np.pi - gaps.max()))
    if kind == "cylinder":
        p["diameter"] = 2 * p["radius"]
    if kind == "sphere":
        p["diameter"] = 2 * p["radius"]
