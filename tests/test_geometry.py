"""Smoke test: scaling/geometry math on synthetic data - no SfM, no Open3D."""

import numpy as np
import pytest

from measure3d import scaling
from mesh_gen.meshing import low_density_mask


def test_compute_scale():
    # picked pair 2 units apart, real distance 210 mm -> 105 mm/unit
    assert scaling.compute_scale(210.0, 2.0) == 105.0


def test_compute_scale_rejects_bad_input():
    with pytest.raises(ValueError):
        scaling.compute_scale(0.0, 2.0)
    with pytest.raises(ValueError):
        scaling.compute_scale(210.0, 0.0)


def test_pair_distances_chunks_pairs_and_ignores_trailing():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 3, 0], [0, 0, 4]], dtype=float)
    picked = [0, 1, 0, 2, 0, 3, 2]  # three pairs + one unpaired index
    assert scaling.pair_distances_mm(points, picked, 10.0) == [10.0, 30.0, 40.0]


def test_scaled_aabb():
    corners = np.array([[0, 0, 0], [1, 2, 3]], dtype=float)
    mn, mx = scaling.scaled_aabb_mm(corners, 100.0)
    assert np.allclose(mn, [0, 0, 0])
    assert np.allclose(mx - mn, [100, 200, 300])
    with pytest.raises(ValueError):
        scaling.scaled_aabb_mm(np.empty((0, 3)), 1.0)


def test_synthetic_object_end_to_end():
    # A 50-unit cube whose real edge is 500 mm: calibrate on one edge, then
    # measurements and the bounding box must come out in true mm.
    rng = np.random.default_rng(42)
    cube = rng.uniform(0.0, 50.0, size=(1000, 3))
    cube[0] = [0.0, 0.0, 0.0]
    cube[1] = [50.0, 0.0, 0.0]  # calibration edge

    picked_units = scaling.distance(cube[0], cube[1])
    scale = scaling.compute_scale(500.0, picked_units)
    assert scale == pytest.approx(10.0)

    mn, mx = scaling.scaled_aabb_mm(cube, scale)
    assert np.all(mx - mn <= 500.0 + 1e-9)
    (measured,) = scaling.pair_distances_mm(cube, [0, 1], scale)
    assert measured == pytest.approx(500.0)


def test_scale_sidecar_roundtrip(tmp_path):
    path = tmp_path / "scale.json"
    scaling.save_scale(path, 105.0, known_distance_mm=210.0, picked_distance_units=2.0)
    assert scaling.load_scale(path) == 105.0
    assert scaling.load_scale(tmp_path / "missing.json") is None


def test_low_density_mask():
    densities = np.arange(100, dtype=float)
    mask = low_density_mask(densities, quantile=0.05)
    assert mask.dtype == bool
    assert mask.sum() == 5
    assert mask[:5].all()
