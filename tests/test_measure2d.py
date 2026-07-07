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
