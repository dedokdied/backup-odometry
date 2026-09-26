"""Fit the pathgraph -> UTM registration from the GNSS tracks and save it.

The supplied pathgraph JSON lives in an unknown local metric frame, while the
jury frame is UTM grid axes zeroed at the start of the recording.  This step
recovers the single rigid transform between the two, once, from the offline GNSS
tracks, and stores it as an artifact so inference never needs the map or GNSS.

Note the cached labels are a *true-north ENU* frame, so they are re-projected
through UTM here rather than merely offset - a pure offset would be wrong by
roughly 120 m over a 5 km route because of the meridian convergence.

    python ml/scripts/12_build_registration.py [--trim 0.8] [--min-duration 200]
"""

from __future__ import annotations

import argparse

import numpy as np

from odom_ml.data.build import HZ, load_labeled, manifest
from odom_ml.geo import enu_to_utm
from odom_ml.position.pathgraph import load_pathgraph
from odom_ml.position.registration import build_registration, save_registration

# Every bag in this dataset sits inside UTM zone 37; forcing it keeps all the
# tracks in one continuous grid instead of splitting at the zone boundary.
UTM_ZONE = 37


def gnss_utm_tracks(min_duration: float, max_per_bag: int = 6000) -> tuple[np.ndarray, int, int]:
    """Stack every bag's rover track into one UTM easting/northing cloud."""
    man = manifest(HZ)
    used = skipped = 0
    chunks: list[np.ndarray] = []
    for rec in man["bags"]:
        if rec["duration"] < min_duration:
            skipped += 1
            continue
        d = load_labeled(rec["bag_id"], HZ)
        if "gnss_lat0" not in d:
            skipped += 1
            continue
        ok = np.isfinite(d["gx"]) & np.isfinite(d["gy"]) & np.isfinite(d["gz"])
        if ok.sum() < 1000:
            skipped += 1
            continue
        enu = np.column_stack([d["gx"][ok], d["gy"][ok], d["gz"][ok]])
        e, n, _ = enu_to_utm(
            enu, float(d["gnss_lat0"]), float(d["gnss_lon0"]), float(d["gnss_alt0"]), zone=UTM_ZONE
        )
        pts = np.column_stack([e, n])
        if pts.shape[0] > max_per_bag:  # keep the cloud manageable, order preserved
            pts = pts[:: int(np.ceil(pts.shape[0] / max_per_bag))]
        chunks.append(pts)
        used += 1
    if not chunks:
        raise SystemExit("no usable GNSS bags - rebuild the cache first")
    return np.vstack(chunks), used, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trim", type=float, default=0.8, help="ICP inlier fraction")
    ap.add_argument("--min-duration", type=float, default=200.0, help="skip shorter bags")
    ap.add_argument("--out", default=None, help="artifact path")
    args = ap.parse_args()

    target, used, skipped = gnss_utm_tracks(args.min_duration)
    print(f"GNSS bags used {used}, skipped {skipped}, points {target.shape[0]}")
    print(
        f"  E {target[:,0].min():.1f}..{target[:,0].max():.1f}  "
        f"N {target[:,1].min():.1f}..{target[:,1].max():.1f}"
    )

    graph = load_pathgraph()
    print(
        f"pathgraph '{graph.name}': {graph.xy.shape[0]} pts, {graph.length:.1f} m, "
        f"x {graph.xy[:,0].min():.1f}..{graph.xy[:,0].max():.1f}, "
        f"y {graph.xy[:,1].min():.1f}..{graph.xy[:,1].max():.1f}"
    )

    reg = build_registration(graph, target, trim=args.trim)
    moved = reg.to_utm(graph.xy)
    print("\nregistration:")
    print(f"  direction       {reg.direction}")
    print(f"  rms residual    {reg.rms_residual_m:.2f} m")
    print(f"  median residual {reg.median_residual_m:.2f} m")
    print(f"  align score     {reg.align_score_m:.2f} m")
    print(f"  inliers <5 m    {reg.inlier_fraction*100:.1f} %")
    print(f"  scale estimate  {reg.scale_estimate:.6f} (expect ~1.0)")
    print(f"  iterations      {reg.iterations}")
    print(f"  rotation        {np.degrees(np.arctan2(reg.rotation[1,0], reg.rotation[0,0])):.4f} deg")
    print(f"  translation     {reg.translation}")
    print(
        f"  pathgraph in UTM: E {moved[:,0].min():.1f}..{moved[:,0].max():.1f}  "
        f"N {moved[:,1].min():.1f}..{moved[:,1].max():.1f}"
    )

    if abs(reg.scale_estimate - 1.0) > 1e-3:
        print("\nWARNING: fitted scale is not ~1, the two frames may not both be metric")
    if reg.median_residual_m > 10.0:
        print("\nWARNING: median residual above 10 m, the registration is probably wrong")

    out = save_registration(reg) if args.out is None else save_registration(reg, args.out)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
