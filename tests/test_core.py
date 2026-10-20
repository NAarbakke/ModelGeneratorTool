"""core: scale math, the calibration sidecar, transforms and the ChArUco board.
Synthetic data only - no SfM, no Open3D."""

import numpy as np
import pytest

from core import board, geometry, scale


# --- scale math and the calibration sidecar ----------------------------------------

def test_compute_scale():
    # picked pair 2 units apart, real distance 210 mm -> 105 mm/unit
    assert scale.compute_scale(210.0, 2.0) == 105.0


def test_compute_scale_rejects_bad_input():
    with pytest.raises(ValueError):
        scale.compute_scale(0.0, 2.0)
    with pytest.raises(ValueError):
        scale.compute_scale(210.0, 0.0)


def test_pair_distances_chunks_pairs_and_ignores_trailing():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 3, 0], [0, 0, 4]], dtype=float)
    picked = [0, 1, 0, 2, 0, 3, 2]  # three pairs + one unpaired index
    assert scale.pair_distances_mm(points, picked, 10.0) == [10.0, 30.0, 40.0]


def test_scaled_aabb():
    corners = np.array([[0, 0, 0], [1, 2, 3]], dtype=float)
    mn, mx = scale.scaled_aabb_mm(corners, 100.0)
    assert np.allclose(mn, [0, 0, 0])
    assert np.allclose(mx - mn, [100, 200, 300])
    with pytest.raises(ValueError):
        scale.scaled_aabb_mm(np.empty((0, 3)), 1.0)


def test_synthetic_object_end_to_end():
    # A 50-unit cube whose real edge is 500 mm: calibrate on one edge, then
    # measurements and the bounding box must come out in true mm.
    rng = np.random.default_rng(42)
    cube = rng.uniform(0.0, 50.0, size=(1000, 3))
    cube[0] = [0.0, 0.0, 0.0]
    cube[1] = [50.0, 0.0, 0.0]  # calibration edge

    picked_units = scale.distance(cube[0], cube[1])
    mm_per_unit = scale.compute_scale(500.0, picked_units)
    assert mm_per_unit == pytest.approx(10.0)

    mn, mx = scale.scaled_aabb_mm(cube, mm_per_unit)
    assert np.all(mx - mn <= 500.0 + 1e-9)
    (measured,) = scale.pair_distances_mm(cube, [0, 1], mm_per_unit)
    assert measured == pytest.approx(500.0)


def test_scale_sidecar_roundtrip(tmp_path):
    path = tmp_path / "scale.json"
    scale.save_scale(path, 105.0, known_distance_mm=210.0, picked_distance_units=2.0)
    assert scale.load_scale(path) == 105.0
    assert scale.load_scale(tmp_path / "missing.json") is None


def test_board_calibration_roundtrip(tmp_path):
    path = tmp_path / "scale.json"
    T = np.eye(4)
    T[:3, :3] = 2.5 * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    T[:3, 3] = [1, 2, 3]
    scale.save_calibration(path, T, "charuco/triangulated-corners", rms_mm=0.03)
    cal = scale.load_calibration(path)
    assert cal.board_frame and cal.info["rms_mm"] == 0.03
    assert cal.scale_mm_per_unit == pytest.approx(2.5)
    assert np.allclose(cal.to_mm([[1, 0, 0]]), [[1, 4.5, 3]])
    assert scale.load_scale(path) == pytest.approx(2.5)


def test_legacy_scale_file_loads_as_pure_scale(tmp_path):
    path = tmp_path / "scale.json"
    path.write_text('{"scale_mm_per_unit": 4.0, "units": "mm"}')
    cal = scale.load_calibration(path)
    assert cal.method == "points" and not cal.board_frame
    assert np.allclose(cal.world_to_mm, np.diag([4.0, 4.0, 4.0, 1.0]))


# --- transforms and the ChArUco board ------------------------------------------------

def _random_similarity(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    T = np.eye(4)
    T[:3, :3] = 37.5 * q
    T[:3, 3] = rng.normal(size=3) * 100
    return T


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    T = _random_similarity(rng)
    src = rng.normal(size=(50, 3))
    dst = src @ T[:3, :3].T + T[:3, 3]
    assert np.allclose(geometry.umeyama(src, dst), T, atol=1e-8)


def test_robust_umeyama_rejects_outliers():
    rng = np.random.default_rng(1)
    T = _random_similarity(rng)
    src = rng.normal(size=(60, 3))
    dst = src @ T[:3, :3].T + T[:3, 3]
    dst[:5] += 50.0  # five gross outliers
    est, keep = geometry.robust_umeyama(src, dst)
    assert not keep[:5].any() and keep[5:].all()
    assert np.allclose(est, T, atol=1e-6)


def test_triangulate_two_views():
    X = np.array([10.0, -5.0, 300.0])
    views = []
    for t in ([0.0, 0.0, 0.0], [-80.0, 0.0, -10.0]):  # camera at -t, looking down +z
        P = np.column_stack([np.eye(3), t])
        cam = P @ np.r_[X, 1]
        views.append((P, cam[:2] / cam[2]))
    est = geometry.triangulate([v[0] for v in views], [v[1] for v in views])
    assert np.allclose(est, X, atol=1e-6)


def test_board_frame_is_centered_and_z_up():
    spec = board.BoardSpec()
    corners = board.to_output_frame(spec, board.board_corners_mm(spec))
    assert np.allclose(corners.mean(0)[:2], 0.0, atol=1e-9)
    assert np.allclose(corners[:, 2], 0.0)
    # proper rotation: x along the board's long side, y flipped, so right-handed with z up
    assert corners[1, 0] - corners[0, 0] == pytest.approx(spec.square_mm)


def test_board_detects_on_its_own_image():
    spec = board.BoardSpec()
    img = board.board_image(spec, px_per_mm=4, margin_mm=10)
    px, ids = board.detect(img, spec)
    n_inner = (spec.squares_x - 1) * (spec.squares_y - 1)
    assert len(ids) == n_inner
    # corner 0 sits one square in from the board origin, plus margin; COLMAP pixel convention
    k = list(ids).index(0)
    assert np.allclose(px[k], [(10 + spec.square_mm) * 4] * 2, atol=0.5)
