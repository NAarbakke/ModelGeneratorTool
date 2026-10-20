"""MCP server exposing the scan-to-CAD harness, so the modelling agent is swappable.

    scan2cad serve <scan>          (stdio; register it with Claude Code, Codex, ...)

The tool logic lives in `Harness` (plain Python, testable without MCP); the
server only wraps it. Candidates live in <scan>/cad/<name>.{py,step,stl,json}.
"""

from __future__ import annotations

import functools
import json
import re
import shutil
from pathlib import Path

import numpy as np
# Module level, not inside build_server: with `from __future__ import annotations` the SDK
# resolves tool return annotations (Image) against module globals.
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from core import workspace as core_ws

from .. import cad
from ..capture import cloud
from ..evaluation import render, scoring
from ..geometry import primitives as prim, segment
from ..workspace import ScanWorkspace

INSTRUCTIONS = """\
You are reverse-engineering a physical part into parametric CAD from a 3D scan.

Conventions
- Units are millimeters. Coordinates are the board frame: Z up, z = 0 is the surface the part
  stands on, the part occupies z >= 0. The underside was never seen by the camera.
- Write CadQuery (`cq` and `math` are preloaded). Assign the final solid to `result`.
- Use the fitted primitives as measurements: a cylinder region with diameter 19.96 mm and axis Z
  is almost certainly a 20 mm hole. Prefer round design values when they fall within the scan
  noise (see noise_floor_mm); keep a measured value when no round value is that close.

Workflow
1. scan_info, then segment (once) and list_regions: what faces exist and how big they are.
2. Build a first model of the overall body, score it, render it. Then add features
   (holes, fillets, chamfers, pockets) one at a time, re-scoring after each.
3. Read score output: scan_to_cad = scan points the model fails to explain (missing
   features); cad_to_scan = model surface no scan supports (invented features);
   per-region deviations and worst areas tell you where to look.
4. Stop when both directions are within tolerance or improvements stall. Report the final
   candidate name, its key dimensions and anything you are unsure about.
"""

NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class Harness:
    def __init__(self, scan: Path) -> None:
        self.ws = ScanWorkspace(Path(scan))
        if not self.ws.object_ply.exists():
            raise FileNotFoundError(f"{self.ws.object_ply} missing - run `scan2cad cloud` first")
        self.points, self.normals = cloud.load_cloud(self.ws.object_ply)
        self._cameras: np.ndarray | None | bool = False  # False = not loaded yet

    # --- state ------------------------------------------------------------------
    @property
    def regions_doc(self) -> dict | None:
        if not self.ws.regions_json.exists():
            return None
        return json.loads(self.ws.regions_json.read_text())

    @property
    def labels(self) -> np.ndarray | None:
        path = self.ws.regions_ply.with_suffix(".labels.npy")
        if not path.exists():
            return None
        labels = np.load(path)
        return labels if len(labels) == len(self.points) else None

    @property
    def tolerance(self) -> float:
        doc = self.regions_doc
        if doc:
            return float(doc["stats"]["tolerance_mm"])
        return max(3 * prim.estimate_noise(self.points), 0.05)

    @property
    def cameras(self) -> np.ndarray | None:
        if self._cameras is False:
            try:
                self._cameras = scoring.camera_centers_mm(self.ws)
            except Exception:  # synthetic/bench scans have no SfM model
                self._cameras = None
        return self._cameras

    def _cand(self, name: str, ext: str) -> Path:
        if not NAME_RE.match(name):
            raise ValueError("candidate names: letters, digits, '_' and '-', at most 64 chars")
        return self.ws.cad_dir / f"{name}.{ext}"

    # --- tools ------------------------------------------------------------------
    def scan_info(self) -> dict:
        lo, hi = self.points.min(0), self.points.max(0)
        meta = self.ws.read_meta()
        info = {
            "points": len(self.points),
            "bbox_min_mm": np.round(lo, 2).tolist(), "bbox_max_mm": np.round(hi, 2).tolist(),
            "extent_mm": np.round(hi - lo, 2).tolist(),
            "scale_source": meta.get("scale_source"), "alignment_rms_mm": meta.get("alignment_rms_mm"),
            "cloud_backend": meta.get("cloud_backend"),
            "noise_sigma_estimate_mm": round(prim.estimate_noise(self.points), 4),
            "tolerance_mm": round(self.tolerance, 4),
            "segmented": self.ws.regions_json.exists(),
            "candidates": sorted(p.stem for p in self.ws.cad_dir.glob("*.step")),
        }
        if info["cloud_backend"] == "hull":
            info["warning"] = ("cloud from the visual hull: top surfaces bulge and pockets/blind "
                               "holes are filled - trust silhouettes and side faces, not top faces")
        return info

    def segment(self, tolerance: float | None = None) -> dict:
        regions, stats = segment.segment(self.points, self.normals, tolerance)
        entries = segment.write_regions(regions, stats, self.points, self.ws.regions_json,
                                        self.ws.regions_ply)
        return {"stats": stats, "regions": [_compact(e) for e in entries]}

    def list_regions(self, kind: str | None = None) -> list[dict]:
        doc = self.regions_doc
        if doc is None:
            raise RuntimeError("not segmented yet - call segment first")
        return [_compact(e) for e in doc["regions"] if kind is None or e["kind"] == kind]

    def region(self, region_id: int) -> dict:
        doc = self.regions_doc
        if doc is None:
            raise RuntimeError("not segmented yet - call segment first")
        if not 0 <= region_id < len(doc["regions"]):
            raise ValueError(f"no region {region_id}")
        return doc["regions"][region_id]

    def fit_region(self, region_id: int, kind: str) -> dict:
        """Force one primitive kind on a region (e.g. test whether a 'freeform' is a cone)."""
        if kind not in prim.KINDS:
            raise ValueError(f"kind must be one of {prim.KINDS}")
        labels = self.labels
        if labels is None:
            raise RuntimeError("not segmented yet - call segment first")
        m = labels == region_id
        f = prim.fit(kind, self.points[m], self.normals[m], self.tolerance)
        if f is None:
            return {"kind": kind, "error": "fit failed (degenerate for this region)"}
        out = f.summary()
        out["within_tolerance"] = bool(f.rms <= self.tolerance * prim.RMS_LIMIT_RATIO)
        return out

    def build(self, name: str, code: str) -> dict:
        res = cad.run_script(code, self.ws.cad_dir, self._cand(name, "py").stem)
        if res.get("ok") and res.get("solids") != 1:
            res["warning"] = f"result has {res.get('solids')} solids; a part should be exactly 1"
        return res

    def score(self, name: str, silhouettes: bool = True) -> dict:
        step = self._cand(name, "step")
        if not step.exists():
            raise FileNotFoundError(f"no candidate {name!r} - build it first")
        mesh = scoring.CadMesh.from_shape(cad.load_step(step))
        res = scoring.deviation(mesh, self.points, self.tolerance, self.cameras, self.labels)
        if silhouettes and self.ws.sfm.has_sparse() and self.ws.masks_dir.exists():
            sil = scoring.silhouettes(self.ws, mesh)
            sil.pop("per_view", None)
            res["silhouettes"] = sil
        if "regions" in res and self.regions_doc:
            kinds = {e["id"]: e["kind"] for e in self.regions_doc["regions"]}
            res["regions"] = {k: {**v, "kind": kinds.get(k)} for k, v in res["regions"].items()
                              if v["mean_mm"] > self.tolerance / 2}  # only the ones worth a look
        self._cand(name, "score.json").write_text(json.dumps(res, indent=1))
        return res

    def render(self, name: str | None = None, views: list[str] | None = None) -> bytes:
        """PNG: CAD shaded, scan points colored by deviation from it (green = within tol)."""
        import cv2

        views = views or ["iso", "iso_back", "top", "front"]
        mesh, colors = None, None
        if name:
            step = self._cand(name, "step")
            if not step.exists():
                raise FileNotFoundError(f"no candidate {name!r}")
            mesh = scoring.CadMesh.from_shape(cad.load_step(step))
            colors = render.deviation_colors(mesh.scene().distance(self.points), self.tolerance)
        sub = self.points[:: max(1, len(self.points) // 60000)]
        sub_c = colors[:: max(1, len(self.points) // 60000)] if colors is not None else None
        imgs = [render.render(v, mesh, sub, sub_c, size=(640, 480)) for v in views]
        png = render.montage(imgs, views)
        render.save_png(png, self.ws.renders_dir / f"{name or 'scan'}.png")
        ok, buf = cv2.imencode(".png", png[..., ::-1])
        return buf.tobytes()

    def show_code(self, name: str) -> str:
        return self._cand(name, "py").read_text()

    def candidates(self) -> list[dict]:
        out = []
        for step in sorted(self.ws.cad_dir.glob("*.step")):
            entry = {"name": step.stem}
            score = step.with_suffix(".score.json")
            if score.exists():
                s = json.loads(score.read_text())
                entry.update(s["summary"], scan_to_cad_mean=s["scan_to_cad"]["mean_mm"],
                             cad_to_scan_mean=s["cad_to_scan"].get("mean_mm"),
                             silhouette_iou=s.get("silhouettes", {}).get("mean_iou"))
            out.append(entry)
        return out

    def import_step(self, name: str, path: str) -> dict:
        """Register an external STEP (e.g. a DualBrep result) as a candidate."""
        target = self._cand(name, "step")
        target.parent.mkdir(parents=True, exist_ok=True)
        shape = cad.load_step(Path(path))
        cad.export_step(shape, target)
        return cad.shape_stats(shape)

    def refit(self, name: str, new_name: str, tolerance: float | None = None) -> dict:
        """Replace spline faces of a candidate with analytic surfaces where they fit."""
        from ..cad import refit as brep_refit

        shape = cad.load_step(self._cand(name, "step"))
        new, reports = brep_refit.refit_shape(shape, tolerance or self.tolerance / 3)
        cad.export_step(new, self._cand(new_name, "step"))
        cad.export_stl(new, self._cand(new_name, "stl"))
        return {**cad.shape_stats(new),
                "faces_replaced": sum(r.replaced for r in reports),
                "faces_kept": [{"face": r.index, "fit": r.fitted, "why": r.note}
                               for r in reports if not r.replaced and r.note != "already analytic"]}

    def export(self, name: str) -> dict:

        step = self._cand(name, "step")
        if not step.exists():
            raise FileNotFoundError(f"no candidate {name!r}")
        out = {"step": str(step)}
        models = core_ws.output_dir(core_ws.OUTPUT_MODELS_DIR)
        if models is not None:
            dest = models / f"{self.ws.name}-{name}.step"
            shutil.copy2(step, dest)
            out["copied_to"] = str(dest)
        return out


def _compact(e: dict) -> dict:
    """Region summary small enough to list dozens of them."""
    out = {"id": e["id"], "kind": e["kind"], "points": e["points"],
           "approx_area_mm2": e["approx_area_mm2"], "centroid": e["centroid"],
           "neighbors": e["neighbors"]}
    f = e.get("fit", {})
    for k in ("rms_mm", "diameter", "radius", "half_angle_deg", "major_radius", "minor_radius",
              "nearest_axis", "axis_angle_deg", "offset", "normal", "axis_dir", "axis_point",
              "apex", "center", "axial_range", "arc_coverage_deg", "also_fits"):
        if k in f:
            out[k] = f[k]
    return out


def _agent_errors(fn):
    """Expected failures (bad name, missing candidate, not segmented yet) reach the agent as
    a readable ToolError instead of the SDK's generic 'Error executing tool'."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, FileNotFoundError, RuntimeError, KeyError) as exc:
            raise ToolError(str(exc)) from exc
    return wrapper


def build_server(scan: Path) -> MCPServer:
    h = Harness(scan)
    server = MCPServer("scan2cad", instructions=INSTRUCTIONS)

    def tool():
        def register(fn):
            return server.tool()(_agent_errors(fn))
        return register

    @tool()
    def scan_info() -> dict:
        """Scan summary: extent, noise, tolerance, scale source, existing candidates."""
        return h.scan_info()

    @tool()
    def segment_scan(tolerance_mm: float | None = None) -> dict:
        """Segment the scan into faces and fit plane/cylinder/cone/sphere/torus to each.
        Default tolerance: 3x the estimated scan noise. Takes a few minutes; run once."""
        return h.segment(tolerance_mm)

    @tool()
    def list_regions(kind: str | None = None) -> list[dict]:
        """Segmented regions (optionally one kind) with their fitted dimensions."""
        return h.list_regions(kind)

    @tool()
    def region(region_id: int) -> dict:
        """Everything known about one region: full fit, bbox, neighbors."""
        return h.region(region_id)

    @tool()
    def fit_region(region_id: int, kind: str) -> dict:
        """Force a primitive kind onto a region (plane, cylinder, cone, sphere, torus)."""
        return h.fit_region(region_id, kind)

    @tool()
    def build_cad(name: str, code: str) -> dict:
        """Run CadQuery code (must assign `result`) as candidate `name`; returns validity,
        volume, bbox and face types, or the error."""
        return h.build(name, code)

    @tool()
    def score_cad(name: str, silhouettes: bool = True) -> dict:
        """Compare candidate `name` with the scan: both deviation directions, per-region
        deviations, worst areas, and silhouette IoU against the object masks."""
        return h.score(name, silhouettes)

    @tool()
    def render_cad(name: str | None = None, views: list[str] | None = None) -> Image:
        """Picture of candidate `name` (or the bare scan) with scan points colored by
        deviation: green within tolerance, yellow, red = 3x tolerance. Views: iso, iso_back,
        top, bottom, front, back, left, right."""
        return Image(data=h.render(name, views), format="png")

    @tool()
    def show_code(name: str) -> str:
        """The CadQuery source of a candidate."""
        return h.show_code(name)

    @tool()
    def list_candidates() -> list[dict]:
        """All candidates with their latest scores."""
        return h.candidates()

    @tool()
    def refit_faces(name: str, new_name: str) -> dict:
        """Replace spline faces of candidate `name` with true planes/cylinders/cones/
        spheres/tori where they fit (for DualBrep or imported STEPs)."""
        return h.refit(name, new_name)

    @tool()
    def export_cad(name: str) -> dict:
        """Mark candidate `name` as the result: copies its STEP to output/models."""
        return h.export(name)

    return server


def serve(scan: Path) -> None:
    build_server(scan).run("stdio")
