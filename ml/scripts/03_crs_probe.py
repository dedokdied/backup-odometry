import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from odom_ml.data import grid_from_bag, read_bag
from odom_ml.geo import latlon_to_utm
from odom_ml.position import load_pathgraph

MAX_SPEED = 40.0


def clean_track(xy, t, alt, max_speed=MAX_SPEED):
    keep = np.ones(xy.shape[0], dtype=bool)
    for _ in range(5):
        d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        dt = np.diff(t)
        sp = np.where(dt > 1e-6, d / np.maximum(dt, 1e-6), 0.0)
        bad = np.zeros(xy.shape[0], dtype=bool)
        idx = np.nonzero(sp > max_speed)[0]
        bad[idx] = True
        bad[idx + 1] = True
        if not bad.any():
            break
        keep &= ~bad
    xy_k, t_k, alt_k = xy[keep], t[keep], alt[keep]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy_k, axis=0), axis=1))])
    return xy_k, alt_k, t_k, s


def subsample(pg, step=1.5):
    ds = float(np.median(np.diff(pg.s)))
    idx = np.unique(np.clip((np.arange(0.0, pg.length, step) / ds).astype(int), 0, pg.s.size - 1))
    return np.column_stack([pg.xy[idx, 0], pg.xy[idx, 1]]), pg.s[idx], pg.z[idx]


def principal_angle(xy):
    c = xy - xy.mean(axis=0)
    w, v = np.linalg.eigh(c.T @ c / c.shape[0])
    return float(np.arctan2(v[1, -1], v[0, -1]))


def nn_dist(a, b):
    d2 = (a[:, None, 0] - b[None, :, 0]) ** 2 + (a[:, None, 1] - b[None, :, 1]) ** 2
    j = np.argmin(d2, axis=1)
    return np.sqrt(d2[np.arange(a.shape[0]), j]), j


def icp(src, dst, theta0, iters=80):
    theta = theta0
    prev = np.inf
    for _ in range(iters):
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        warped = (rot @ src.T).T
        d, j = nn_dist(warped, dst)
        matched = dst[j]
        sc = src - src.mean(axis=0)
        mc = matched.mean(axis=0)
        dc = matched - mc
        a = float(np.sum(sc[:, 0] * dc[:, 0] + sc[:, 1] * dc[:, 1]))
        b = float(np.sum(sc[:, 0] * dc[:, 1] - sc[:, 1] * dc[:, 0]))
        theta = float(np.arctan2(b, a))
        err = float(np.mean(d))
        if abs(prev - err) < 1e-5:
            break
        prev = err
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return theta, (rot @ src.T).T, float(prev)


def main(bag_id, path_file=None):
    pg = load_pathgraph(Path(path_file) if path_file else None)
    pxy, ps, pz = subsample(pg)
    pxy = pxy - pxy.mean(axis=0)

    raw = read_bag(bag_id)
    df = grid_from_bag(raw, hz=10.0).frame
    m = np.isfinite(df["mlat"].to_numpy()) & np.isfinite(df["mlon"].to_numpy())
    lat = df.loc[m, "mlat"].to_numpy()
    lon = df.loc[m, "mlon"].to_numpy()
    alt = df.loc[m, "malt"].to_numpy()
    t = df.loc[m, "t"].to_numpy()
    e, n, _ = latlon_to_utm(lat, lon)
    gxy = np.column_stack([e, n])
    gxy, alt, t, s = clean_track(gxy, t, alt)
    gxy = gxy - gxy.mean(axis=0)
    print(f"{bag_id}: fixes={m.sum()} kept={gxy.shape[0]} traj_len={s[-1]:.1f} m")
    print(f"  gnss  E[{e.min():.0f}..{e.max():.0f}] N[{n.min():.0f}..{n.max():.0f}] alt[{alt.min():.1f}..{alt.max():.1f}]")
    print(f"  gnss bbox {gxy[:,0].min():.0f}..{gxy[:,0].max():.0f} / {gxy[:,1].min():.0f}..{gxy[:,1].max():.0f}  ang={np.rad2deg(principal_angle(gxy)):.2f}")
    print(f"  path bbox {pxy[:,0].min():.0f}..{pxy[:,0].max():.0f} / {pxy[:,1].min():.0f}..{pxy[:,1].max():.0f}  ang={np.rad2deg(principal_angle(pxy)):.2f}")

    stride = max(1, gxy.shape[0] // 600)
    src = gxy[::stride]
    best = None
    for dth in np.arange(-4.0, 4.01, 0.5):
        th, warped, err = icp(src, pxy, principal_angle(pxy) - principal_angle(src) + np.deg2rad(dth))
        if best is None or err < best[2]:
            best = (th, warped, err)
    theta, warped_src, err = best
    d, j = nn_dist((lambda c, s: (np.array([[c, -s], [s, c]]) @ warped_src.T).T)(np.cos(theta), np.sin(theta)), pxy)
    print(f"\nICP best: theta={np.rad2deg(theta):+.3f} deg mean_nn={err:.2f} m")
    print(f"  dist to path: median={np.median(d):.2f} p90={np.percentile(d,90):.2f} max={d.max():.2f} m")
    print(f"  matched arclength: [{ps[j].min():.1f}..{ps[j].max():.1f}] of {pg.length:.1f}")
    zd = alt[::stride] - pz[j]
    print(f"  gnss_alt - path_z: mean={zd.mean():+.2f} std={zd.std():.2f}")
    print(f"  path_frame = R({np.rad2deg(theta):+.3f}) @ (utm - gnss_centroid) + path_centroid_offset")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "30618_0e41eac3", sys.argv[2] if len(sys.argv) > 2 else None)
