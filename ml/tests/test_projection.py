import numpy as np
import pytest

from odom_ml.position.pathgraph import PathGraph
from odom_ml.position.projection import MapProjection
from odom_ml.position.registration import RouteRegistration, registration_path


def _straight_route(length=1000.0, step=1.0, offset=0.0, northing=6100000.0):
    n = int(length / step) + 1
    s = np.arange(n, dtype=np.float64) * step
    xy = np.column_stack([offset + s, np.full(n, northing)])
    tang = np.zeros(n)
    return PathGraph("straight", xy, np.zeros(n), tang, np.zeros(n), s)


def _rot(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _identity_reg(rot=None, trans=(0.0, 0.0)) -> RouteRegistration:
    rot = np.eye(2) if rot is None else rot
    return RouteRegistration(
        rotation=rot,
        translation=np.asarray(trans, dtype=np.float64),
        direction="forward",
        rms_residual_m=0.0,
        median_residual_m=0.0,
        inlier_fraction=1.0,
        scale_estimate=1.0,
        iterations=1,
        n_correspondences=0,
        align_score_m=0.0,
        source_name="synthetic",
        notes="",
    )


def test_projection_recovers_arclength_and_zero_lateral():
    g = _straight_route()
    proj = MapProjection(g, _identity_reg())
    out = proj.project_utm(g.xy[:, 0], g.xy[:, 1])
    np.testing.assert_allclose(out.s, g.s, atol=1e-6)
    np.testing.assert_allclose(out.lateral, 0.0, atol=1e-6)


def test_lateral_sign_is_positive_to_the_left_of_travel():
    g = _straight_route()
    proj = MapProjection(g, _identity_reg())
    # +N is to the left when heading +E, so a point north of the track is positive
    out = proj.project_utm(np.array([500.0]), np.array([6100000.0 + 3.0]))
    assert out.s[0] == pytest.approx(500.0, abs=1e-6)
    assert out.lateral[0] == pytest.approx(3.0, abs=1e-6)
    left = proj.project_utm(np.array([500.0]), np.array([6100000.0 - 3.0]))
    assert left.lateral[0] == pytest.approx(-3.0, abs=1e-6)


def test_projection_snaps_onto_the_nearest_segment_not_the_nearest_vertex():
    g = _straight_route(step=1.0)
    proj = MapProjection(g, _identity_reg())
    out = proj.project_utm(np.array([500.4]), np.array([6100000.0]))
    assert out.s[0] == pytest.approx(500.4, abs=1e-6)
    assert abs(out.lateral[0]) < 1e-9


def test_projection_follows_the_registration_rotation():
    g = _straight_route()
    rot = _rot(30.0)
    trans = np.array([1000.0, -2000.0])
    proj = MapProjection(g, _identity_reg(rot, trans))
    utm = g.xy @ rot.T + trans
    out = proj.project_utm(utm[:, 0], utm[:, 1])
    np.testing.assert_allclose(out.s, g.s, atol=1e-6)
    np.testing.assert_allclose(out.lateral, 0.0, atol=1e-6)


def test_tangent_is_rotated_into_utm():
    g = _straight_route()
    rot = _rot(90.0)
    proj = MapProjection(g, _identity_reg(rot))
    tan = proj.tangent_at(np.array([0.0, 10.0]))
    # the route runs along +x in the map frame, which is +y (north) in UTM
    np.testing.assert_allclose(tan, np.array([[0.0, 1.0], [0.0, 1.0]]), atol=1e-9)


def test_s_to_utm_inverts_projection():
    g = _straight_route()
    rot = _rot(-15.0)
    proj = MapProjection(g, _identity_reg(rot, [5.0, 7.0]))
    s = np.linspace(0.0, g.length, 51)
    e, n = proj.s_to_utm(s)
    back = proj.project_utm(e, n)
    np.testing.assert_allclose(back.s, s, atol=1e-6)
    np.testing.assert_allclose(back.lateral, 0.0, atol=1e-6)


def test_output_frame_is_utm_minus_the_starting_point():
    g = _straight_route()
    proj = MapProjection(g, _identity_reg())
    e, n = proj.s_to_utm(np.array([0.0, 100.0]))
    out = proj.s_to_output(np.array([0.0, 100.0]), origin=(e[0], n[0]))
    np.testing.assert_allclose(out, np.array([[0.0, 0.0], [100.0, 0.0]]), atol=1e-9)


def test_to_output_frame_subtracts_every_axis():
    out = MapProjection.to_output(
        np.array([400000.0]), np.array([6100000.0]), np.array([180.0]), (399000.0, 6090000.0, 150.0)
    )
    np.testing.assert_allclose(out, np.array([[1000.0, 10000.0, 30.0]]))


def test_ambiguity_flags_a_self_overlapping_route():
    # out and back along the same line: every point is close to two others
    s = np.arange(0.0, 1001.0, 1.0)
    xy = np.column_stack([np.concatenate([s, s]), np.zeros(s.size * 2)])
    s2 = np.concatenate([s, s + 1000.0])
    g = PathGraph("loopback", xy, np.zeros(s2.size), np.zeros(s2.size), np.zeros(s2.size), s2)
    amb = MapProjection(g, _identity_reg()).ambiguity()
    assert amb["max_arclength_separation_m"] > 500.0


def test_ambiguity_is_small_for_an_out_and_back_on_widely_separated_tracks():
    # the real route returns on a separate track, 20 m away, so the two legs are
    # further apart than the 15 m ambiguity radius
    g = _straight_route()
    e, n = g.xy[:, 0], g.xy[:, 1] + 20.0
    out = np.concatenate([g.xy, np.column_stack([e[::-1][1:], n[::-1][1:]])])
    s2 = np.concatenate([g.s, g.s[-1] + g.s[::-1][1:]])
    g2 = PathGraph("twotrack", out, np.zeros(s2.size), np.zeros(s2.size), np.zeros(s2.size), s2)
    amb = MapProjection(g2, _identity_reg()).ambiguity()
    assert amb["max_arclength_separation_m"] < 30.0


def test_ds_is_the_arclength_increment():
    g = _straight_route()
    proj = MapProjection(g, _identity_reg())
    out = proj.project_utm(g.xy[:, 0], g.xy[:, 1])
    assert np.isnan(out.ds[0])
    np.testing.assert_allclose(out.ds[1:], 1.0, atol=1e-9)
    assert out.travelled() == pytest.approx(g.length)


def test_sampling_rate_invariance_of_travelled_distance():
    g = _straight_route()
    proj = MapProjection(g, _identity_reg())
    e, n = proj.s_to_utm(g.s)
    fine = proj.project_utm(e, n)
    coarse = proj.project_utm(e[::10], n[::10])
    assert coarse.travelled() == pytest.approx(fine.travelled(), rel=1e-9)


def test_real_route_projection_is_self_consistent():
    """The shipped route must survive its own registration and projection."""
    if not registration_path().exists():
        pytest.skip("route_registration.json not built")
    mp = MapProjection.load()
    rep = mp.sanity()
    assert rep["s_roundtrip_max_err_m"] < 1e-6
    assert rep["lateral_max_abs_m"] < 1e-6
    # the two legs are on separate tracks, so a nearest-point arclength is unique
    assert rep["max_arclength_separation_m"] < 30.0
    assert rep["length_m"] > 4000.0


def test_real_route_tangent_matches_the_json_tang_field():
    if not registration_path().exists():
        pytest.skip("route_registration.json not built")
    mp = MapProjection.load()
    s = np.linspace(0.0, mp.graph.length, 200)
    got = mp.tangent_at(s)
    # rotate the JSON tang field into UTM at the same arclengths and compare
    ref_t = np.interp(s, mp.graph.s, mp.graph.tang)
    ref = np.column_stack([np.cos(ref_t), np.sin(ref_t)]) @ mp.reg.rotation.T
    ang = np.arctan2(np.sin(got[:, 1] * ref[:, 0] - got[:, 0] * ref[:, 1]), (got * ref).sum(axis=1))
    assert np.median(np.abs(ang)) < 0.02
    assert np.max(np.abs(ang)) < 0.1
