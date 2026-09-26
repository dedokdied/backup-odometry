"""The exported route and reference trajectory must be mutually consistent.

A wrong transform here is invisible in the file itself -- the numbers look
perfectly reasonable -- and only shows up as a position estimate that is
thousands of kilometres wrong.  The cross-track check below is the guard: it
compares the reference trajectory against the route it claims to follow.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from odom_ml.export import to_array
from odom_ml.position.projection import MapProjection

ROUTE = Path("artifacts/route/route_utm.json")
TRUTH_DIR = Path("artifacts/reference")


def _load(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"{path} not generated; run ml/scripts/23_export_route.py")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def route() -> dict:
    return _load(ROUTE)


@pytest.fixture(scope="module")
def route_arrays(route) -> dict[str, np.ndarray]:
    p = np.asarray(route["points"], dtype=np.float64)
    names = route["columns"]
    return {n: p[:, i] for i, n in enumerate(names)}


def test_route_covers_the_whole_alignment(route_arrays):
    assert route_arrays["s"].size == 4710
    assert route_arrays["s"][0] == 0.0
    assert route_arrays["s"][-1] == pytest.approx(4708.3, abs=0.5)
    assert np.all(np.diff(route_arrays["s"]) > 0), "arclength must increase"


def test_northing_has_no_false_northing(route_arrays):
    """UTM northings here are ~6.19e6, not ~16.19e6.

    Mixing the two conventions is a 10 000 km error, and the artefact has to say
    which one it uses without the reader having to infer it.
    """
    n = route_arrays["n"]
    assert 6.0e6 < n.mean() < 7.0e6, f"unexpected northing range {n.min()}..{n.max()}"
    assert "false_northing" in json.loads(ROUTE.read_text(encoding="utf-8"))["conventions"]


def test_tangents_and_normals_are_orthonormal_unit_vectors(route_arrays):
    t = np.column_stack([route_arrays["t_e"], route_arrays["t_n"]])
    nvec = np.column_stack([route_arrays["n_e"], route_arrays["n_n"]])
    np.testing.assert_allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(nvec, axis=1), 1.0, atol=1e-9)
    # left normal: rotating the tangent by +90 deg
    np.testing.assert_allclose(nvec[:, 0], -t[:, 1], atol=1e-9)
    np.testing.assert_allclose(nvec[:, 1], t[:, 0], atol=1e-9)


def test_route_reprojects_onto_itself_in_utm(route_arrays):
    """Feeding the exported UTM back through the projection must recover s."""
    mp = MapProjection.load()
    proj = mp.project_utm(route_arrays["e"], route_arrays["n"])
    assert np.abs(proj.s - route_arrays["s"]).max() < 1.0
    assert np.abs(proj.lateral).max() < 1.0


@pytest.mark.parametrize("path", sorted(TRUTH_DIR.glob("ground_truth_*.json")))
def test_ground_truth_actually_follows_the_route(path):
    """The guard against a double transform.

    A reference trajectory produced by pushing the GNSS through the registration
    as well as through the geodetic conversion lands thousands of kilometres away
    and still serialises perfectly.  Its cross-track against the route does not.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = doc["rows"]
    org = doc["conventions"]["origin_utm"]
    e = to_array([r[1] for r in rows]) + org[0]
    n = to_array([r[2] for r in rows]) + org[1]
    ct = to_array([r[5] for r in rows])
    on = np.array([r[6] for r in rows], dtype=bool)
    assert on.any(), "no on-route rows at all"

    assert np.abs(ct[on]).max() < 50.0, (
        f"on-route rows are up to {np.abs(ct[on]).max():.0f} m from the corridor"
    )
    assert np.median(np.abs(ct[on])) < 10.0
    # off-route rows are the depot and should genuinely be far away
    if (~on).any():
        assert np.abs(ct[~on]).min() > 15.0, "rows marked off-route are on the corridor"

    s = to_array([r[4] for r in rows])
    assert s[on].max() - s[on].min() > 1000.0, "run does not traverse the route"
