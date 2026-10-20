"""Slower end-to-end checks of segmentation and B-rep refitting (minutes). Skip: -m "not slow"."""

import numpy as np
import pytest

from scan2cad import cad
from scan2cad.cad import refit as brep_refit
from scan2cad.evaluation import scoring
from scan2cad.geometry import segment
from scan2cad.synthetic.parts import PARTS

pytestmark = pytest.mark.slow


def _shape(tmp_path, name):
    res = cad.run_script(PARTS[name], tmp_path, name)
    assert res["ok"], res
    return cad.load_step(tmp_path / f"{name}.step")


def test_segment_block_hole_finds_every_visible_face(tmp_path):
    shape = _shape(tmp_path, "block_hole")
    mesh = scoring.CadMesh.from_shape(shape, 0.01)
    pts, nrm = mesh.sample(30000)
    keep = pts[:, 2] > 0.75  # the bottom face is never seen
    pts = pts[keep] + np.random.default_rng(0).normal(0, 0.1, (keep.sum(), 3))
    regions, stats = segment.segment(pts, nrm[keep])
    assert stats["noise_sigma_estimate_mm"] == pytest.approx(0.1, abs=0.02)
    assert stats["kinds"]["plane"] == 5 and stats["kinds"]["cylinder"] == 5
    radii = sorted(r.fit.params["radius"] for r in regions if r.kind == "cylinder")
    assert radii[:4] == pytest.approx([6, 6, 6, 6], abs=0.15)  # corner fillets
    assert radii[4] == pytest.approx(10, abs=0.1)  # the hole


@pytest.mark.parametrize("name", ["block_hole", "flange", "bracket"])
def test_refit_restores_analytic_faces(tmp_path, name):
    gt = _shape(tmp_path, name)
    splines = brep_refit.to_splines(gt)
    assert set(cad.shape_stats(splines)["faces"]) == {"bspline"}
    new, reports = brep_refit.refit_shape(splines, tolerance=0.05)
    stats = cad.shape_stats(new)
    assert stats["valid"] and stats["solids"] == 1
    assert stats["faces"] == cad.shape_stats(gt)["faces"]
    assert cad.volume_iou(gt, new) > 0.998
