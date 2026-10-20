"""scan2cad unit tests on synthetic data: no video, no SfM, no SAM 2 model download."""

import json

import numpy as np
import pytest

from scan2cad import cad
from scan2cad.capture import frames
from scan2cad.evaluation import scoring
from scan2cad.geometry import raycast
from scan2cad.synthetic.parts import PARTS


# --- frame selection -----------------------------------------------------------------

def test_shots_split_at_cuts():
    cut = np.zeros(100)
    cut[[30, 70]] = 0.9
    assert frames.shots(cut) == [(0, 30), (30, 70), (70, 100)]


def test_select_frames_prefers_sharp_and_drops_blurry():
    sharp = np.full(100, 100.0)
    sharp[::10] = 150.0  # the sharpest frame of each 10-frame bin
    sharp[50:60] = 5.0   # a blurry stretch: its best frame is still far below the median
    picks = frames.select_frames(sharp, 0, 100, 10)
    assert picks == [0, 10, 20, 30, 40, 60, 70, 80, 90]


# --- CAD kernel ---------------------------------------------------------------------

def test_run_script_ok_and_errors(tmp_path):
    ok = cad.run_script(PARTS["block_hole"], tmp_path, "ok")
    assert ok["ok"] and ok["valid"] and ok["solids"] == 1
    assert ok["faces"] == {"plane": 6, "cylinder": 5}
    assert ok["bbox_size_mm"] == pytest.approx([80, 50, 30], abs=1e-3)
    assert (tmp_path / "ok.step").exists() and (tmp_path / "ok.stl").exists()

    missing = cad.run_script("x = 1", tmp_path, "missing")
    assert not missing["ok"] and "result" in missing["error"]
    broken = cad.run_script("result = cq.Workplane().box(1, 1, 1).fillet(5)", tmp_path, "broken")
    assert not broken["ok"]


def test_volume_iou(tmp_path):
    import cadquery as cq

    a = cq.Workplane().box(10, 10, 10).val().wrapped
    b = cq.Workplane().box(10, 10, 10).translate((5, 0, 0)).val().wrapped
    assert cad.volume_iou(a, a) == pytest.approx(1.0)
    assert cad.volume_iou(a, b) == pytest.approx(1 / 3, abs=0.003)  # 500 / 1500


# --- scoring -------------------------------------------------------------------------

def test_deviation_prefers_the_right_model(tmp_path):
    import cadquery as cq

    gt = cq.Workplane().box(40, 30, 20, centered=(True, True, False)) \
        .faces(">Z").workplane().hole(10).val().wrapped
    wrong = cq.Workplane().box(40, 30, 20, centered=(True, True, False)).val().wrapped
    rng = np.random.default_rng(0)
    pts, _ = scoring.CadMesh.from_shape(gt).sample(20000)
    pts = pts[pts[:, 2] > 0.75] + rng.normal(0, 0.05, (np.sum(pts[:, 2] > 0.75), 3))
    good = scoring.deviation(scoring.CadMesh.from_shape(gt), pts, 0.15)
    bad = scoring.deviation(scoring.CadMesh.from_shape(wrong), pts, 0.15)
    assert good["scan_to_cad"]["mean_mm"] < 0.06
    assert good["summary"]["f_score"] > 0.99
    assert bad["summary"]["f_score"] < good["summary"]["f_score"]
    # the missing hole shows up as an unexplained cluster on the axis
    worst = bad["worst_scan_areas"][0]["centroid"]
    assert np.hypot(worst[0], worst[1]) < 2.0


# --- object cloud: mask votes ----------------------------------------------------------

class _PinholeCam:
    """Stand-in for pycolmap.Camera.img_from_cam."""

    def __init__(self, K):
        self.K = K

    def img_from_cam(self, cam):
        uv = cam @ self.K.T
        return uv[:, :2] / uv[:, 2:]


def test_mask_vote_keeps_object_points():
    from scan2cad.capture.cloud import View, vote_filter

    K = raycast.intrinsics(400, 200, 200)
    views = []
    for az in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        eye = 300 * np.array([np.cos(az), np.sin(az), 0.5])
        R, t = raycast.look_at(eye, [0, 0, 0])
        cam = raycast.PinholeCamera(K, R, t, 200, 200)
        # mask = projection of a sphere of radius 20 at the origin (disc around the center)
        uv, _ = cam.project(np.zeros((1, 3)))
        yy, xx = np.mgrid[0:200, 0:200] + 0.5
        r_px = 400 * 20 / np.linalg.norm(eye)
        mask = (xx - uv[0, 0]) ** 2 + (yy - uv[0, 1]) ** 2 < r_px ** 2
        views.append(View(_PinholeCam(K), np.column_stack([R, t]), mask))
    inside = np.random.default_rng(0).normal(0, 5, (200, 3))
    outside = inside + [60, 0, 0]
    keep = vote_filter(np.vstack([inside, outside]), views)
    assert keep[:200].mean() > 0.95 and keep[200:].mean() < 0.05


# --- DualBrep adapter -------------------------------------------------------------------

def test_dualbrep_normalization_round_trip(tmp_path):
    import cadquery as cq
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf, gp_Vec

    from scan2cad.cad import dualbrep

    part = cq.Workplane().box(80, 50, 30, centered=(True, True, False)).translate((5, -3, 0))
    shape = part.val().wrapped
    pts, nrm = scoring.CadMesh.from_shape(shape).sample(20000)
    bundle = dualbrep.prepare(pts, nrm, tmp_path / "bundle")
    norm = json.loads((bundle / "normalization.json").read_text())
    import open3d as o3d

    cloud = np.asarray(o3d.io.read_point_cloud(str(bundle / "sample" / "scan.ply")).points)
    assert len(cloud) == dualbrep.NUM_POINTS
    assert np.max(np.abs(cloud)) == pytest.approx(dualbrep.HALF_EXTENT, abs=0.01)

    # pretend DualBrep returned the shape perfectly, in its normalized frame
    fwd = gp_Trsf()
    fwd.SetTranslation(gp_Vec(*(-np.asarray(norm["center_mm"]))))
    s = gp_Trsf()
    s.SetScaleFactor(norm["scale_per_mm"])
    s.Multiply(fwd)
    cad.export_step(BRepBuilderAPI_Transform(shape, s, True).Shape(), bundle / "scan.step")
    back = dualbrep.import_step(bundle)
    assert cad.volume_iou(shape, back) > 0.999
    assert cad.shape_stats(back)["bbox_size_mm"] == pytest.approx([80, 50, 30], abs=0.01)


# --- MCP harness (no MCP transport) -------------------------------------------------------

def test_harness_build_score_and_names(tmp_path):
    from scan2cad.agent import bench
    from scan2cad.agent.server import Harness

    ws = bench.make_quick_case("block_hole", tmp_path / "case", noise_mm=0.05, points=20000)
    h = Harness(ws.root)
    info = h.scan_info()
    assert info["extent_mm"] == pytest.approx([80, 50, 29.25], abs=0.5)
    assert info["noise_sigma_estimate_mm"] == pytest.approx(0.05, abs=0.02)

    assert h.build("v1", PARTS["block_hole"])["valid"]
    s = h.score("v1", silhouettes=False)
    assert s["summary"]["f_score"] > 0.98
    assert h.candidates()[0]["name"] == "v1"
    png = h.render("v1", ["iso"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(ValueError):
        h.build("../evil", "result = cq.Workplane().box(1, 1, 1)")
    ev = bench.evaluate(h.ws.cad_dir / "v1.step", ws.root / "gt" / "part.step")
    assert ev["volume_iou"] > 0.999 and ev["face_types_match"]
