"""Replace spline faces of a B-rep with true analytic surfaces where they fit.

DualBrep (and any scan-to-B-rep tool) represents every face, even flat ones, as
a B-spline. For each face: sample it, run `primitives.fit_best`, and if an
analytic model fits within tolerance, rebuild the face on that surface.

Scope - read before relying on it: the new face keeps the original boundary
edges and ShapeFix recomputes their parametric curves on the new surface. That
is sound when the fit deviation is below the edge tolerance (faces that are
analytic up to noise). It does NOT re-intersect neighbouring surfaces to compute
new exact edges. Each replacement is validated - BRepCheck validity AND volume
IoU >= MIN_IOU against the input solid, since a solid can pass BRepCheck yet break
booleans - and rolled back otherwise, so the output is never worse than the input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import primitives as prim

MIN_IOU = 0.995


@dataclass
class FaceReport:
    index: int
    original: str
    fitted: str | None
    rms_mm: float | None
    replaced: bool
    note: str = ""


def face_samples(face, grid: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Points and oriented (outward) normals on a face, from a UV grid clipped to the face.

    UV sampling, not the triangulation: a flat spline face triangulates into a
    handful of corner nodes, far too few to fit.
    """
    from OCP.BRepGProp import BRepGProp_Face
    from OCP.BRepTools import BRepTools
    from OCP.BRepTopAdaptor import BRepTopAdaptor_FClass2d
    from OCP.gp import gp_Pnt, gp_Pnt2d, gp_Vec
    from OCP.TopAbs import TopAbs_IN

    umin, umax, vmin, vmax = BRepTools.UVBounds_s(face)
    classifier = BRepTopAdaptor_FClass2d(face, 1e-6)
    prop = BRepGProp_Face(face)  # Normal() accounts for the face orientation
    pts, nrm = [], []
    for u in np.linspace(umin, umax, grid):
        for v in np.linspace(vmin, vmax, grid):
            if classifier.Perform(gp_Pnt2d(u, v)) != TopAbs_IN:
                continue
            pp, vv = gp_Pnt(), gp_Vec()
            prop.Normal(u, v, pp, vv)
            if vv.Magnitude() < 1e-12:
                continue
            pts.append([pp.X(), pp.Y(), pp.Z()])
            nrm.append(np.array([vv.X(), vv.Y(), vv.Z()]) / vv.Magnitude())
    return np.asarray(pts).reshape(-1, 3), np.asarray(nrm).reshape(-1, 3)


def analytic_surface(f: prim.Fit, face_points: np.ndarray | None = None, pole: str = "side",
                     seam_point: np.ndarray | None = None):
    """OCC Geom surface for a primitive fit.

    Closed surfaces (cylinder, cone, sphere, torus) have a parametric seam along
    their reference X direction. If the seam runs through the face, the face's
    boundary self-intersects in UV and the face is invalid, so X is aimed directly
    away from the face's points: the seam then lies on the side the face doesn't cover.
    """
    from OCP.Geom import (Geom_ConicalSurface, Geom_CylindricalSurface, Geom_Plane,
                          Geom_SphericalSurface, Geom_ToroidalSurface)
    from OCP.gp import gp_Ax3, gp_Dir, gp_Pnt

    p = f.params

    def ax3(origin, direction):
        origin, direction = np.asarray(origin, float), prim._unit(np.asarray(direction, float))
        if face_points is None or f.kind == "plane":
            return gp_Ax3(gp_Pnt(*origin), gp_Dir(*direction))
        if seam_point is not None:  # put the seam on the face's existing seam edge
            radial = np.asarray(seam_point) - origin
            sign = 1.0
        else:  # put the seam opposite the face
            radial = (face_points - origin).mean(0)
            sign = -1.0
        radial = radial - (radial @ direction) * direction
        if np.linalg.norm(radial) < 1e-9:  # full revolution: any seam position works
            return gp_Ax3(gp_Pnt(*origin), gp_Dir(*direction))
        return gp_Ax3(gp_Pnt(*origin), gp_Dir(*direction), gp_Dir(*(sign * prim._unit(radial))))

    if f.kind == "plane":
        return Geom_Plane(ax3(p["point"], p["normal"]))
    if f.kind == "cylinder":
        return Geom_CylindricalSurface(ax3(p["axis_point"], p["axis_dir"]), p["radius"])
    if f.kind == "sphere":
        return Geom_SphericalSurface(ax3(p["center"], _sphere_pole(p, face_points, pole)),
                                     p["radius"])
    if f.kind == "torus":
        return Geom_ToroidalSurface(ax3(p["center"], p["axis_dir"]),
                                    p["major_radius"], p["minor_radius"])
    if f.kind == "cone":
        # OCC cones are defined by a reference circle; put it 1 unit along the axis from the apex.
        a = np.asarray(p["axis_dir"])
        half = np.radians(p["half_angle_deg"])
        return Geom_ConicalSurface(ax3(np.asarray(p["apex"]) + a, a), half, float(np.tan(half)))
    raise ValueError(f.kind)


def _sphere_pole(p: dict, face_points: np.ndarray | None, pole: str) -> np.ndarray:
    """Pole direction: "side" (90 degrees off the face center: keeps pole and seam off a
    small patch) or "center" (through the face center: right for a dome, whose seam and
    degenerate pole ShapeFix then adds)."""
    if face_points is None:
        return np.array([0.0, 0.0, 1.0])
    d = face_points.mean(0) - np.asarray(p["center"])
    if np.linalg.norm(d) < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    d = prim._unit(d)
    return d if pole == "center" else prim._basis(d)[0]


def seam_midpoint(face) -> np.ndarray | None:
    """Midpoint of the face's seam edge (an edge closed on the face), if it has one."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    exp = TopExp_Explorer(face, TopAbs_EDGE)
    while exp.More():
        edge = TopoDS.Edge_s(exp.Current())
        if BRep_Tool.IsClosed_s(edge, face) and not BRep_Tool.Degenerated_s(edge):
            c = BRepAdaptor_Curve(edge)
            pt = c.Value(0.5 * (c.FirstParameter() + c.LastParameter()))
            return np.array([pt.X(), pt.Y(), pt.Z()])
        exp.Next()
    return None


def surface_variants(f: prim.Fit, face_points: np.ndarray, face):
    """Candidate surfaces to try in turn: seam away from the face; for faces that
    already carry a seam edge, seam on that edge; for spheres, both pole placements."""
    yield analytic_surface(f, face_points)
    seam = seam_midpoint(face) if f.kind != "plane" else None
    if seam is not None:
        yield analytic_surface(f, face_points, seam_point=seam)
    if f.kind == "sphere":
        yield analytic_surface(f, face_points, pole="center")
        if seam is not None:
            yield analytic_surface(f, face_points, pole="center", seam_point=seam)


def rebuild_face(face, surface, tolerance: float, outward: tuple[np.ndarray, np.ndarray]):
    """The face's boundary wires on a new surface; None if no valid face results.

    The new surface's natural normal may point the other way from the old one's,
    which flips how the old wires wind relative to it, so both wire directions
    are tried. If the old outer wire carries seam or degenerate (pole) edges that
    only make sense on the old surface, a wire of just the real boundary edges is
    tried too, and ShapeFix adds the new surface's own seam and pole. The result
    is then oriented to match `outward` (points, normals).
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepTools import BRepTools
    from OCP.ShapeFix import ShapeFix_Face
    from OCP.TopAbs import TopAbs_WIRE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    outer = BRepTools.OuterWire_s(face)
    inner = []
    exp = TopExp_Explorer(face, TopAbs_WIRE)
    while exp.More():
        w = TopoDS.Wire_s(exp.Current())
        if not w.IsSame(outer):
            inner.append(w)
        exp.Next()

    outers = [outer]
    real = _real_boundary_wire(face, outer)
    if real is not None:
        outers.append(real)
    for base, flip in ((o, fl) for o in outers for fl in (False, True)):
        def wire(w):
            return TopoDS.Wire_s(w.Reversed()) if flip else w

        mk = BRepBuilderAPI_MakeFace(surface, wire(base), True)
        if not mk.IsDone():
            continue
        for w in inner:
            mk.Add(wire(w))
        fixer = ShapeFix_Face(mk.Face())
        fixer.SetPrecision(tolerance)
        fixer.SetMaxTolerance(4 * tolerance)
        fixer.FixAddNaturalBoundMode = False
        fixer.Perform()
        new = fixer.Face()
        if not BRepCheck_Analyzer(new).IsValid():
            continue
        x, n = face_samples(new, grid=12)
        if len(x) == 0:
            continue
        if _normal_agreement(x, n, *outward) < 0:
            new = TopoDS.Face_s(new.Reversed())
        return new
    return None


def _real_boundary_wire(face, outer):
    """The outer wire without seam/degenerate edges, or None if it has none or the
    remaining edges don't form one connected wire (e.g. a through-hole's two circles)."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeWire
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    mk, dropped = BRepBuilderAPI_MakeWire(), False
    exp = TopExp_Explorer(outer, TopAbs_EDGE)
    while exp.More():
        edge = TopoDS.Edge_s(exp.Current())
        if BRep_Tool.IsClosed_s(edge, face) or BRep_Tool.Degenerated_s(edge):
            dropped = True
        else:
            mk.Add(edge)
        exp.Next()
    return mk.Wire() if dropped and mk.IsDone() else None


def _normal_agreement(x: np.ndarray, n: np.ndarray, ref_x: np.ndarray, ref_n: np.ndarray) -> float:
    """Mean dot product between normals at x and the nearest reference normals."""
    from scipy.spatial import cKDTree

    _, i = cKDTree(ref_x).query(x)
    return float(np.mean(np.sum(n * ref_n[i], axis=1)))


def refit_shape(shape, tolerance: float = 0.1, kinds: tuple[str, ...] = prim.KINDS):
    """Return (new_shape, reports). Faces already analytic are left alone."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepTools import BRepTools_ReShape
    from OCP.ShapeFix import ShapeFix_Shape
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    from . import SURFACE_TYPES, volume_iou

    reports: list[FaceReport] = []
    current = shape
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    faces = []
    while exp.More():
        faces.append(TopoDS.Face_s(exp.Current()))
        exp.Next()

    for i, face in enumerate(faces):
        kind = SURFACE_TYPES.get(BRepAdaptor_Surface(face).GetType().name, "other")
        if kind in prim.KINDS:
            reports.append(FaceReport(i, kind, kind, None, False, "already analytic"))
            continue
        x, n = face_samples(face)
        if len(x) < 20:
            reports.append(FaceReport(i, kind, None, None, False, "too few samples"))
            continue
        best, _ = prim.fit_best(x, n, tolerance, kinds)
        if best is None:
            reports.append(FaceReport(i, kind, None, None, False, "no analytic fit: kept spline"))
            continue
        new_face = next((nf for surf in surface_variants(best, x, face)
                         if (nf := rebuild_face(face, surf, tolerance, (x, n))) is not None), None)
        if new_face is None:
            reports.append(FaceReport(i, kind, best.kind, best.rms, False, "rebuild failed"))
            continue
        # ReShape composes the new face's orientation with the old occurrence's, which can
        # leave a face inside-out in the shell; the IoU check catches that, so also try it reversed.
        accepted, note = None, ""
        for oriented in (new_face, TopoDS.Face_s(new_face.Reversed())):
            reshape = BRepTools_ReShape()
            reshape.Replace(face, oriented)
            fix = ShapeFix_Shape(reshape.Apply(current))
            fix.SetPrecision(tolerance)
            fix.SetMaxTolerance(4 * tolerance)
            fix.Perform()
            candidate = fix.Shape()
            if not BRepCheck_Analyzer(candidate).IsValid():
                note = "solid invalid after replacement: kept spline"
                continue
            iou = volume_iou(shape, candidate, samples=400_000)
            if iou >= MIN_IOU:
                accepted = candidate
                break
            note = f"volume IoU vs input {iou:.4f} after replacement: kept spline"
        if accepted is None:
            reports.append(FaceReport(i, kind, best.kind, best.rms, False, note))
            continue
        current = accepted
        reports.append(FaceReport(i, kind, best.kind, best.rms, True))
    return current, reports


def to_splines(shape):
    """Convert every face to a B-spline (used to test refit, mimics DualBrep output)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert

    conv = BRepBuilderAPI_NurbsConvert(shape, True)
    return conv.Shape()
