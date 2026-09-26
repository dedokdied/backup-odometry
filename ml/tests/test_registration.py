"""Tests for the pathgraph -> UTM rigid registration.

The synthetic cases pin the row-vector rotation convention and the centred-frame
translation bookkeeping, which are both easy to get subtly wrong and produce a
plausible-looking but completely mis-registered map.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from conftest import reference_bags, stack_track_utm

from odom_ml.position.registration import (
    FORWARD,
    REVERSE,
    RouteRegistration,
    build_registration,
    fit_registration,
    load_registration,
    register_pathgraph,
    save_registration,
)


def curve(n=600, seed=0, noise=0.5):
    """A curved, non-degenerate 2-D track."""
    t = np.linspace(0.0, 1.0, n)
    x = 1000.0 * t
    y = 300.0 * np.sin(2.0 * np.pi * t) + 50.0 * t**2
    pts = np.column_stack([x, y])
    if noise:
        pts = pts + np.random.default_rng(seed).normal(0.0, noise, (n, 2))
    return pts


def rigid(theta_deg, t):
    a = np.deg2rad(theta_deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]), np.asarray(t, float)


def test_recovers_exact_transform():
    src = curve()
    rot, t = rigid(37.0, [401000.0, 6188000.0])
    dst = src @ rot.T + t

    fit = fit_registration(src, dst, trim=0.9, subsample=len(src))
    assert np.allclose(fit.rotation, rot, atol=1e-8)
    assert np.allclose(fit.translation, t, atol=1e-6)
    assert fit.rms_residual_m < 1e-6


def test_rotation_is_a_proper_rotation():
    src = curve()
    rot, t = rigid(-118.0, [400000.0, 6187000.0])
    fit = fit_registration(src, src @ rot.T + t, trim=0.9, subsample=len(src))
    assert np.linalg.det(fit.rotation) == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(fit.rotation @ fit.rotation.T, np.eye(2), atol=1e-9)
    assert fit.scale_estimate == pytest.approx(1.0, abs=1e-3)


def test_tolerates_noise_and_outliers():
    src = curve()
    rot, t = rigid(12.0, [400500.0, 6187500.0])
    dst = src @ rot.T + t
    # 5 % of the target is garbage, as a baggy GNSS track would be
    rng = np.random.default_rng(1)
    bad = rng.choice(dst.shape[0], dst.shape[0] // 20, replace=False)
    dst = dst.copy()
    dst[bad] = rng.uniform([398000.0, 6186000.0], [404000.0, 6189000.0], (bad.size, 2))

    fit = fit_registration(src, dst, trim=0.8, subsample=len(src))
    angle = np.degrees(np.arctan2(fit.rotation[1, 0], fit.rotation[0, 0]))
    assert angle % 360.0 == pytest.approx(12.0, abs=0.5)
    assert fit.translation == pytest.approx(t, abs=5.0)
    assert fit.median_residual_m < 5.0


def test_inverse_transform_round_trips():
    src = curve()
    rot, t = rigid(64.0, [402000.0, 6189000.0])
    dst = src @ rot.T + t
    reg = fit_registration(src, dst, trim=0.95, subsample=len(src))

    assert np.allclose(reg.to_utm(src), dst, atol=1e-6)
    assert np.allclose(reg.to_pathgraph(dst[:, 0], dst[:, 1]), src, atol=1e-6)


def test_to_local_is_utm_minus_origin():
    reg = RouteRegistration(rotation=np.eye(2), translation=np.array([400000.0, 6188000.0]))
    out = reg.to_local(np.array([400010.0, 400020.0]), np.array([6188010.0, 6188030.0]), 400000.0, 6188000.0)
    assert np.allclose(out, [[10.0, 10.0], [20.0, 30.0]])


def test_picks_the_matching_direction_of_an_out_and_back_route():
    # A path that goes out along +x and comes back along -x: the reversed
    # correspondence must be recognised, not fitted as a 180 deg rotation.
    out = curve(300)
    line = np.vstack([out, out[::-1] + np.array([0.0, 40.0])])
    rot, t = rigid(-25.0, [400900.0, 6187600.0])
    dst = line @ rot.T + t
    dst = dst + np.random.default_rng(2).normal(0.0, 0.3, dst.shape)

    reg = register_pathgraph(line, dst, trim=0.8)
    assert reg.direction in (FORWARD, REVERSE)
    assert reg.median_residual_m < 5.0
    assert abs(reg.scale_estimate - 1.0) < 1e-2


def test_json_artifact_round_trip(tmp_path):
    src = curve()
    rot, t = rigid(15.0, [400700.0, 6187800.0])
    reg = fit_registration(src, src @ rot.T + t, trim=0.9, subsample=len(src))
    reg.source_name = "test"
    reg.notes = "hello"

    path = save_registration(reg, tmp_path / "reg.json")
    back = load_registration(path)
    assert np.allclose(back.rotation, reg.rotation)
    assert np.allclose(back.translation, reg.translation)
    assert back.direction == reg.direction
    assert back.source_name == "test"
    assert back.notes == "hello"
    assert back.scale_estimate == pytest.approx(reg.scale_estimate)


def test_rejects_degenerate_input():
    with pytest.raises(ValueError):
        fit_registration(np.zeros((2, 2)), np.zeros((2, 2)))


# --- the real question: does the registration hold up on runs it never saw? ---


@pytest.mark.slow
def test_held_out_runs_are_registered_to_sub_metre_accuracy(projection_train_only, test_split):
    """Fit on training runs, score on the runs held out from the fit.

    This is the only test that says anything about registration accuracy.  A
    transform fitted on the same bags it is measured on reports the residual of
    its own fit, so it would stay green even if the whole route were displaced.
    """
    target = stack_track_utm(test_split)
    reg = projection_train_only.reg
    moved = reg.to_utm(projection_train_only.graph.xy)
    dist = cKDTree(target).query(moved)[0]

    assert dist.size == moved.shape[0]
    assert np.median(dist) < 1.0, f"median {np.median(dist):.2f} m, worst {dist.max():.2f} m"
    assert np.percentile(dist, 95) < 3.0
    assert (dist < 5.0).mean() > 0.99, f"only {(dist < 5.0).mean()*100:.1f}% within 5 m"


@pytest.mark.slow
def test_registration_scale_is_one_on_real_runs(projection_train_only):
    """Both frames are metric, so a fitted scale far from 1 means a units bug."""
    s = projection_train_only.reg.scale_estimate
    assert abs(s - 1.0) < 5e-4, f"scale estimate {s:.6f}"


@pytest.mark.slow
def test_registration_is_stable_across_fit_subsets(train_split, pathgraph):
    """Two disjoint halves of the training runs must agree on the transform.

    A fit that depended on which runs went in would be fitting noise.
    """
    half = len(train_split) // 2
    fits = [
        build_registration(pathgraph, stack_track_utm(part), trim=0.8)
        for part in (train_split[:half], train_split[half:])
    ]
    d = np.degrees(
        np.arctan2(
            fits[0].rotation[1, 0] * fits[1].rotation[0, 0]
            - fits[0].rotation[0, 0] * fits[1].rotation[1, 0],
            fits[0].rotation[0, 0] * fits[1].rotation[0, 0]
            + fits[0].rotation[1, 0] * fits[1].rotation[1, 0],
        )
    )
    assert abs(d) < 0.05, f"halves disagree on rotation by {d:.4f} deg"
    offset = np.linalg.norm(fits[0].translation - fits[1].translation)
    assert offset < 3.0, f"halves disagree on translation by {offset:.2f} m"


@pytest.mark.slow
def test_saved_registration_matches_a_fresh_fit(projection):
    """The shipped artifact must be what the current code produces from the data.

    Otherwise the artifact is stale and every downstream number is unverifiable.
    """
    target = stack_track_utm(reference_bags("master")[:30])
    fresh = build_registration(projection.graph, target, trim=0.8)
    d = np.degrees(
        np.arctan2(
            projection.reg.rotation[1, 0] * fresh.rotation[0, 0]
            - projection.reg.rotation[0, 0] * fresh.rotation[1, 0],
            projection.reg.rotation[0, 0] * fresh.rotation[0, 0]
            + projection.reg.rotation[1, 0] * fresh.rotation[1, 0],
        )
    )
    assert abs(d) < 0.05, f"saved artifact is {d:.4f} deg away from a fresh fit"
    assert np.linalg.norm(projection.reg.translation - fresh.translation) < 5.0
