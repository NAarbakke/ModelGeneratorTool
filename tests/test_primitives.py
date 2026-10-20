"""Primitive fitting on synthetic patches with known parameters and 0.1 mm noise."""

import numpy as np
import pytest

from scan2cad.geometry import primitives as prim

NOISE = 0.1
TOL = 0.3


def _noisy(x, n, rng):
    return x + rng.normal(0, NOISE, x.shape), n


def plane_patch(rng, m=1500):
    normal = prim._unit(np.array([0.2, -0.1, 1.0]))
    u, v = prim._basis(normal)
    uv = rng.uniform(-30, 30, (m, 2))
    x = np.array([5.0, 3.0, 12.0]) + uv[:, :1] * u + uv[:, 1:] * v
    return _noisy(x, np.tile(normal, (m, 1)), rng)


def cylinder_patch(rng, r=10.0, arc=np.pi / 2, m=1500):
    """A quarter cylinder (a fillet strip) along a tilted axis."""
    a = prim._unit(np.array([0.0, 0.3, 1.0]))
    u, v = prim._basis(a)
    t = rng.uniform(0, arc, m)
    h = rng.uniform(0, 40, m)
    n = np.cos(t)[:, None] * u + np.sin(t)[:, None] * v
    x = np.array([1.0, 2.0, 3.0]) + h[:, None] * a + r * n
    return _noisy(x, n, rng)


def sphere_patch(rng, r=9.0, m=1500):
    d = rng.normal(size=(m, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d[:, 2] = np.abs(d[:, 2])  # upper hemisphere only
    return _noisy(np.array([0.0, 0.0, 28.0]) + r * d, d, rng)


def cone_patch(rng, half_angle_deg=45.0, m=1500):
    """A 45 degree chamfer band, axis Z, apex below."""
    al = np.radians(half_angle_deg)
    apex = np.array([0.0, 0.0, -19.0])
    h = rng.uniform(19.0, 22.0, m)
    t = rng.uniform(0, 2 * np.pi, m)
    radial = np.column_stack([np.cos(t), np.sin(t), np.zeros(m)])
    x = apex + h[:, None] * np.array([0, 0, 1.0]) + (h * np.tan(al))[:, None] * radial
    n = np.cos(al) * radial - np.sin(al) * np.array([0, 0, 1.0])
    return _noisy(x, n, rng)


def torus_patch(rng, R=15.0, r=7.0, m=2000):
    """The outer-top quarter of a torus: a convex fillet on a round edge."""
    t = rng.uniform(0, 2 * np.pi, m)
    p = rng.uniform(0, np.pi / 2, m)
    radial = np.column_stack([np.cos(t), np.sin(t), np.zeros(m)])
    n = np.cos(p)[:, None] * radial + np.sin(p)[:, None] * np.array([0, 0, 1.0])
    x = np.array([0, 0, 21.0]) + R * radial + r * n
    return _noisy(x, n, rng)


@pytest.fixture
def rng():
    return np.random.default_rng(1)


def test_plane(rng):
    best, _ = prim.fit_best(*plane_patch(rng), TOL)
    assert best.kind == "plane"
    assert abs(best.params["normal"] @ prim._unit(np.array([0.2, -0.1, 1.0]))) > 0.9999
    assert best.rms == pytest.approx(NOISE, abs=0.02)


def test_quarter_cylinder(rng):
    best, _ = prim.fit_best(*cylinder_patch(rng), TOL)
    assert best.kind == "cylinder"
    assert best.params["radius"] == pytest.approx(10.0, abs=0.1)
    assert best.params["arc_coverage_deg"] == pytest.approx(90, abs=15)


def test_sphere(rng):
    best, _ = prim.fit_best(*sphere_patch(rng), TOL)
    assert best.kind == "sphere"
    assert best.params["radius"] == pytest.approx(9.0, abs=0.05)
    assert np.allclose(best.params["center"], [0, 0, 28], atol=0.1)


def test_cone(rng):
    best, _ = prim.fit_best(*cone_patch(rng), TOL)
    assert best.kind == "cone"
    assert best.params["half_angle_deg"] == pytest.approx(45.0, abs=1.0)
    assert best.params["nearest_axis"] == "Z" and best.params["axis_angle_deg"] < 1.0


def test_torus(rng):
    best, _ = prim.fit_best(*torus_patch(rng), TOL)
    assert best.kind == "torus"
    assert best.params["major_radius"] == pytest.approx(15.0, abs=0.15)
    assert best.params["minor_radius"] == pytest.approx(7.0, abs=0.15)


def test_freeform_region_is_rejected(rng):
    # A saddle z = (x^2 - y^2)/40: no analytic primitive fits it to 0.3 mm.
    xy = rng.uniform(-25, 25, (1500, 2))
    z = (xy[:, 0] ** 2 - xy[:, 1] ** 2) / 40
    x = np.column_stack([xy, z])
    n = np.column_stack([-xy[:, 0] / 20, xy[:, 1] / 20, np.ones(len(xy))])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    best, fits = prim.fit_best(x, n, TOL)
    assert best is None
    assert fits  # fits were attempted and reported


def test_axis_snap():
    assert prim.axis_snap_deg(np.array([0.0, 0.0, -1.0])) == ("Z", 0.0)
    axis, ang = prim.axis_snap_deg(np.array([1.0, 0.0, 0.05]))
    assert axis == "X" and ang == pytest.approx(np.degrees(np.arctan(0.05)), abs=1e-6)
