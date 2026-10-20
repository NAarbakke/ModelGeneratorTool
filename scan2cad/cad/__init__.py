"""CAD kernel helpers: CadQuery for modelling, OCP directly for checks and export.

Submodules: runner (subprocess entry point), refit (spline faces -> analytic),
dualbrep (DualBrep bridge).

Model-written CadQuery runs in a subprocess (`run_script`) with a timeout, so a
script that loops or segfaults the kernel can't take the MCP server down. All
lengths are mm.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT_TIMEOUT_S = 120

SURFACE_TYPES = {  # GeomAbs_SurfaceType enum names -> short labels
    "GeomAbs_Plane": "plane", "GeomAbs_Cylinder": "cylinder", "GeomAbs_Cone": "cone",
    "GeomAbs_Sphere": "sphere", "GeomAbs_Torus": "torus", "GeomAbs_BezierSurface": "bezier",
    "GeomAbs_BSplineSurface": "bspline", "GeomAbs_SurfaceOfRevolution": "revolution",
    "GeomAbs_SurfaceOfExtrusion": "extrusion", "GeomAbs_OffsetSurface": "offset",
    "GeomAbs_OtherSurface": "other",
}


def to_shape(obj):
    """Accept a cq.Workplane, cq.Shape or raw TopoDS_Shape; return a TopoDS_Shape."""
    import cadquery as cq

    if isinstance(obj, cq.Workplane):
        vals = [v for v in obj.vals() if isinstance(v, cq.Shape)]
        if not vals:
            raise ValueError("Workplane holds no shapes")
        obj = vals[0] if len(vals) == 1 else cq.Compound.makeCompound(vals)
    if isinstance(obj, cq.Shape):
        return obj.wrapped
    return obj


def load_step(path: Path):
    import cadquery as cq

    return cq.importers.importStep(str(path)).val().wrapped


def export_step(shape, path: Path) -> None:
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Interface import Interface_Static
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = STEPControl_Writer()
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    writer.Transfer(shape, STEPControl_AsIs)
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"STEP export failed: {path}")


def tessellate(shape, tolerance: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Triangle mesh (vertices mm, triangle indices) of a shape."""
    import cadquery as cq

    verts, tris = cq.Shape.cast(shape).tessellate(tolerance, 0.2)
    return (np.array([v.toTuple() for v in verts], dtype=float),
            np.array(tris, dtype=np.int64).reshape(-1, 3))


def export_stl(shape, path: Path, tolerance: float = 0.05) -> None:
    import trimesh

    v, f = tessellate(shape, tolerance)
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(v, f, process=False).export(path)


def face_types(shape) -> dict[str, int]:
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    counts: dict[str, int] = {}
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        kind = BRepAdaptor_Surface(TopoDS.Face_s(exp.Current())).GetType().name
        label = SURFACE_TYPES.get(kind, kind)
        counts[label] = counts.get(label, 0) + 1
        exp.Next()
    return counts


def shape_stats(shape) -> dict:
    """Validity, solid count, volume, area, bbox and face-type histogram."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    if shape is None or shape.IsNull():
        return {"valid": False, "solids": 0, "volume_mm3": 0.0, "area_mm2": 0.0,
                "bbox_min_mm": None, "bbox_size_mm": None, "faces": {}}
    vol, area = GProp_GProps(), GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, vol)
    BRepGProp.SurfaceProperties_s(shape, area)
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)  # exact; Add_s pads curved faces
    x0, y0, z0, x1, y1, z1 = box.Get() if not box.IsVoid() else (0.0,) * 6
    solids, exp = 0, TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        solids += 1
        exp.Next()
    return {
        "valid": bool(BRepCheck_Analyzer(shape).IsValid()),
        "solids": solids,
        "volume_mm3": round(vol.Mass(), 3),
        "area_mm2": round(area.Mass(), 3),
        "bbox_min_mm": [round(v, 3) for v in (x0, y0, z0)],
        "bbox_size_mm": [round(v, 3) for v in (x1 - x0, y1 - y0, z1 - z0)],
        "faces": face_types(shape),
    }


def volume_iou(a, b, samples: int = 2_000_000, seed: int = 0) -> float:
    """Volumetric IoU of two solids by Monte Carlo inside/outside tests on their meshes.

    Not OCCT booleans: two solids with coincident analytic faces but different
    edges (e.g. ground truth vs a refitted scan) can make Common return an empty
    shape with no error. Sampling is robust; the standard error is about
    sqrt(IoU * (1 - IoU) / samples_in_union), i.e. ~5e-4 at the default count.
    """
    import open3d as o3d

    meshes = [tessellate(s, 0.02) for s in (a, b)]
    lo = np.minimum(meshes[0][0].min(0), meshes[1][0].min(0))
    hi = np.maximum(meshes[0][0].max(0), meshes[1][0].max(0))
    pts = np.random.default_rng(seed).uniform(lo, hi, (samples, 3)).astype(np.float32)
    inside = []
    for v, f in meshes:
        scene = o3d.t.geometry.RaycastingScene()
        m = o3d.t.geometry.TriangleMesh()
        m.vertex.positions = o3d.core.Tensor(v.astype(np.float32))
        m.triangle.indices = o3d.core.Tensor(f.astype(np.uint32))
        scene.add_triangles(m)
        inside.append(scene.compute_occupancy(o3d.core.Tensor(pts), nsamples=3).numpy() > 0.5)
    union = np.logical_or(*inside).sum()
    return float(np.logical_and(*inside).sum() / union) if union else 0.0


def run_script(source: str, out_dir: Path, name: str,
               timeout: float = SCRIPT_TIMEOUT_S) -> dict:
    """Run CadQuery source in a subprocess; it must assign `result`.

    Writes <name>.py/.step/.stl/.json into out_dir and returns the stats dict
    (or {"ok": False, "error": ...}).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    script = out_dir / f"{name}.py"
    script.write_text(source, encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "scan2cad.cad.runner", str(script), str(out_dir / name)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"script timed out after {timeout:.0f} s"}
    report = out_dir / f"{name}.json"
    if proc.returncode != 0 or not report.exists():
        err = (proc.stderr or proc.stdout).strip().splitlines()
        return {"ok": False, "error": "\n".join(err[-15:]) or f"exit code {proc.returncode}"}
    return json.loads(report.read_text())
