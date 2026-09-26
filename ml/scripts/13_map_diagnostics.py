"""Measure how well the registered route constrains the vehicle.

The position output can only be as good as the route model, so before building
an estimator this measures the two things that decide its design:

* is the route effectively one-dimensional?  A vehicle confined to a known
  polyline has a single unknown (arclength) and its lateral offset is pure
  observation noise.  If the lateral offset were not small, the route would not
  be a usable position reference.
* what is the noise floor?  The spread of the projected arclength while the
  vehicle is standing still bounds how precisely the map can localise it.

It also reports the wheel-speed scale against the route, because the arclength
is the only channel through which speed error reaches the position output.

    python ml/scripts/13_map_diagnostics.py [--limit 20] [--min-duration 200]
"""

from __future__ import annotations

import argparse

import numpy as np

from odom_ml.data.build import HZ, load_labeled, manifest
from odom_ml.geo import enu_to_utm
from odom_ml.position.projection import MapProjection

UTM_ZONE = 37
STOP_SPEED = 0.3  # m/s, below this the vehicle counts as standing still
MOVE_SPEED = 1.0  # m/s, above this a sample is used for the scale fit
ON_ROUTE_M = 15.0  # m, beyond this the vehicle is outside the mapped corridor
MIN_STOP = 25  # samples (0.5 s at 50 Hz) for a stop to be usable
SMOOTH_S = 2.0  # s, window for de-noising arclength before differentiating


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win < 2:
        return x
    k = np.ones(win) / win
    pad = win // 2
    xp = np.pad(np.nan_to_num(x), (pad, pad), mode="edge")
    out = np.convolve(xp, k, mode="valid")[: x.size]
    return out


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs as ``[start, stop)`` index pairs."""
    d = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def robust_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope through the origin, with a 3-sigma trim."""
    if x.size < 10:
        return float("nan")
    k = float((x * y).sum() / (x * x).sum())
    for _ in range(3):
        r = y - k * x
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        keep = np.abs(r) < 3.0 * s
        if keep.sum() < 10:
            break
        k = float((x[keep] * y[keep]).sum() / (x[keep] ** 2).sum())
    return k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all bags")
    ap.add_argument("--min-duration", type=float, default=200.0)
    args = ap.parse_args()

    mp = MapProjection.load()
    print("route sanity:")
    for k, v in mp.sanity().items():
        print(f"  {k:<30} {v:.4f}" if isinstance(v, float) else f"  {k:<30} {v}")

    man = manifest(HZ)
    bags = [b for b in man["bags"] if b["duration"] >= args.min_duration]
    if args.limit:
        bags = bags[: args.limit]

    rows = []
    for rec in bags:
        d = load_labeled(rec["bag_id"], HZ)
        if "gnss_lat0" not in d:
            continue
        ok = np.isfinite(d["gx"]) & np.isfinite(d["gy"]) & np.isfinite(d["gz"])
        if ok.sum() < 1000:
            continue
        enu = np.column_stack([d["gx"][ok], d["gy"][ok], d["gz"][ok]])
        e, n, _ = enu_to_utm(
            enu, float(d["gnss_lat0"]), float(d["gnss_lon0"]), float(d["gnss_alt0"]), zone=UTM_ZONE
        )
        t, v = d["t"][ok], d["speed"][ok]
        dt = float(np.median(np.diff(t)))
        p = mp.project_utm(e, n)
        s = p.s
        if not np.all(np.isfinite(s)):
            continue
        lat = np.abs(p.lateral)

        on = lat < ON_ROUTE_M
        off = runs(~on)
        # a stop is only a noise measurement if the vehicle is on the mapped route
        stops = [
            (a, b)
            for a, b in runs((v < STOP_SPEED) & on)
            if b - a >= MIN_STOP
        ]
        jit = [float(s[a:b].std()) for a, b in stops]

        # arclength advance vs wheel speed, both restricted to the mapped route
        win = max(3, int(round(SMOOTH_S / dt)) | 1)
        s_s = moving_average(s, win)
        v_path = np.abs(np.diff(s_s)) / dt
        m = (v[:-1] > MOVE_SPEED) & (v[1:] > MOVE_SPEED) & on[:-1] & on[1:]
        rows.append(
            {
                "bag": rec["bag_id"],
                "n": int(s.size),
                "dur": float(t[-1] - t[0]),
                "on_frac": float(on.mean()),
                "off_head": float(off[0][1] / len(s)) if off and off[0][0] == 0 else 0.0,
                "off_tail": float((len(s) - off[-1][0]) / len(s)) if off and off[-1][1] == len(s) else 0.0,
                "n_off": len(off),
                "off_lat": float(np.median(lat[~on])) if (~on).any() else 0.0,
                "s0": float(s[0]),
                "s1": float(s[-1]),
                "net": float(s[-1] - s[0]),
                "lat_med": float(np.median(lat[on])),
                "lat_p95": float(np.percentile(lat[on], 95)) if on.any() else float("nan"),
                "jit_med": float(np.median(jit)) if jit else float("nan"),
                "jit_p90": float(np.percentile(jit, 90)) if jit else float("nan"),
                "n_stops": len(jit),
                "scale": robust_slope(v[:-1][m], v_path[m]),
            }
        )

    def col(k):
        return np.array([r[k] for r in rows], dtype=np.float64)

    def clean(k):
        v = col(k)
        return v[np.isfinite(v)]

    print(f"\n{len(rows)} bags, {col('n').sum():.0f} samples, "
          f"{col('dur').sum()/3600:.1f} h\n")

    on = col("on_frac")
    never = int((on <= 0.0).sum())
    print("route coverage (is the mapped corridor actually where the vehicle drives?):")
    print(f"  on-route time            median {np.median(on)*100:.1f} %, "
          f"p10 {np.percentile(on,10)*100:.1f} %, min {on.min()*100:.1f} %")
    print(f"  off-route time           median {(1-np.median(on))*100:.1f} %")
    print(f"  off-route at the start   {(col('off_head') > 0).sum()}/{len(rows)} bags, "
          f"median {np.median(col('off_head'))*100:.1f} % of the recording")
    print(f"  off-route at the end     {(col('off_tail') > 0).sum()}/{len(rows)} bags, "
          f"median {np.median(col('off_tail'))*100:.1f} %")
    mid = int((col("n_off") > 0).sum() - ((col("off_head") > 0) | (col("off_tail") > 0)).sum())
    print(f"  off-route in the middle  {mid}/{len(rows)} bags  <- must be 0 for a single corridor")
    print(f"  never enters the corridor {never} bag(s)")
    print(f"  distance when off-route  median {np.median(col('off_lat')):.0f} m from the corridor")

    print("\nroute as a position reference (on-route samples only):")
    print(f"  |lateral| median of bag medians  {np.median(clean('lat_med')):.2f} m")
    print(f"  |lateral| median of bag p95     {np.median(clean('lat_p95')):.2f} m")
    print(f"  fleet arclength span             {np.min(col('s0')):.0f}..{np.max(col('s1')):.0f} m "
          f"of {mp.graph.length:.0f} m")
    print(f"  |net s| / full route length      median "
          f"{np.median(np.abs(col('net')))/mp.graph.length:.3f}"
          f" (1.0 = traversed the whole corridor)")

    print("\nnoise floor, from the arclength jitter while standing still on the route:")
    jm, jp = clean("jit_med"), clean("jit_p90")
    print(f"  usable stops {jm.size}/{len(rows)} bags, "
          f"{int(col('n_stops').sum())} stop segments total")
    print(f"  within-stop s jitter  median of medians {np.median(jm):.3f} m, "
          f"median of p90 {np.median(jp):.3f} m, worst bag p90 {np.nanmax(jp):.2f} m")
    print("  (this is the along-track resolution of the map; the cross-track offset")
    print("   above does not limit the arclength, it only shows the pathgraph is not")
    print("   exactly the driven line)")

    print("\nwheel speed vs arclength advance (how speed error reaches the position):")
    sc = clean("scale")
    print(f"  scale median {np.median(sc):.5f}, p5 {np.percentile(sc,5):.5f}, "
          f"p95 {np.percentile(sc,95):.5f}")
    for pct in (5, 50, 95):
        k = np.percentile(sc, pct) - 1.0
        print(f"    p{pct:<3} bias {k*100:+.3f} %  ->  {k*mp.graph.length:+7.1f} m over one route")
    spread = np.percentile(sc, 95) - np.percentile(sc, 5)
    print(f"  spread between bags (p95-p5) {spread*100:.3f} % "
          f"-> {spread*mp.graph.length:.1f} m if left uncorrected")


if __name__ == "__main__":
    main()
