"""Smoke test: measure2d planar homography math on synthetic data - no OpenCV/GUI."""

import numpy as np
import pytest

from measure2d import geometry


def synthetic_camera() -> np.ndarray:
    # A plausible perspective warp of a tabletop plane (mm -> image pixels).
    return np.array(
        [
            [0.9, 0.12, 40.0],
            [-0.08, 1.05, 25.0],
            [2e-4, -1.5e-4, 1.0],
        ]
    )


def test_homography_recovers_in_plane_distances():
    metric_corners = geometry.rect_corners(210.0, 297.0)  # A4
    H_true = synthetic_camera()
    image_corners = geometry.apply_homography(H_true, metric_corners)

    H = geometry.solve_homography(image_corners, metric_corners)

    # A 100 mm segment somewhere else on the plane, seen through the same camera.
    segment_mm = np.array([[15.0, 30.0], [115.0, 30.0]])
    segment_px = geometry.apply_homography(H_true, segment_mm)
    recovered = geometry.apply_homography(H, segment_px)
    assert np.linalg.norm(recovered[1] - recovered[0]) == pytest.approx(100.0, abs=1e-6)


def test_homography_roundtrip_corners():
    corners_px = np.array([[100, 80], [900, 120], [880, 700], [90, 650]], dtype=float)
    metric = geometry.rect_corners(210.0, 297.0)
    H = geometry.solve_homography(corners_px, metric)
    assert np.allclose(geometry.apply_homography(H, corners_px), metric, atol=1e-8)


def test_homography_rejects_collinear_corners():
    collinear = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
    with pytest.raises(np.linalg.LinAlgError):
        geometry.solve_homography(collinear, geometry.rect_corners(10.0, 10.0))


def test_reference_presets():
    assert geometry.REFERENCE_PRESETS["a4"] == (210.0, 297.0)
    assert geometry.REFERENCE_PRESETS["credit-card"] == (85.6, 53.98)
    # corners come back in TL, TR, BR, BL click order
    assert np.allclose(
        geometry.rect_corners(2.0, 3.0), [[0, 0], [2, 0], [2, 3], [0, 3]]
    )


def test_fine_pick_recovers_full_res_point_subpixel():
    """The zoom stage must localize a target to ~1/zoom of a full-res pixel,
    regardless of how coarse the first click was."""
    from measure2d import gui

    W, H, crop, zoom = 4000, 3000, gui.FINE_CROP, gui.FINE_ZOOM
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(500):
        target = rng.uniform([crop, crop], [W - crop, H - crop])
        coarse = target + rng.uniform(-crop / 2 + 4, crop / 2 - 4, 2)  # rough click, still in patch
        origin = gui.fine_crop_origin(coarse, crop, W, H)
        # user clicks the fine pixel nearest the target (integer display pixel)
        fine = np.round((target - origin) * zoom)
        recovered = gui.fine_to_full(origin, fine, zoom)
        worst = max(worst, np.linalg.norm(np.subtract(recovered, target)))
    assert worst <= np.sqrt(2) * 0.5 / zoom + 1e-9  # <= half a fine-pixel each axis


def test_fine_crop_origin_clamps_to_image():
    from measure2d import gui

    W, H, crop = 4000, 3000, 160
    # near a corner: origin must not go negative or exceed the image
    ox, oy = gui.fine_crop_origin((5, 5), crop, W, H)
    assert ox == 0 and oy == 0
    ox, oy = gui.fine_crop_origin((W, H), crop, W, H)
    assert ox == W - crop and oy == H - crop
