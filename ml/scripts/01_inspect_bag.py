import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from odom_ml import config as C
from odom_ml.data import grid_from_bag, read_bag

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

bag_id = sys.argv[1] if len(sys.argv) > 1 else "30618_0e41eac3"
ts = None
raw = read_bag(bag_id, typestore=ts)
print(f"bag={bag_id} vehicle={raw.vehicle} t0_ns={raw.t0_ns}")
for topic in C.ALL_TOPICS:
    t = raw.t[topic]
    if t.size == 0:
        print(f"  {topic:36s} EMPTY")
        continue
    rel = (t - raw.t0_ns) / 1e9
    stamp = (raw.header_stamp[topic] - raw.t0_ns) / 1e9
    vals = raw.values[topic]
    print(
        f"  {topic:36s} n={t.size:7d} hz={t.size / max(rel[-1], 1e-9):6.2f} "
        f"t=[{rel[0]:8.2f},{rel[-1]:8.2f}] stamp-t=[{np.min(stamp - rel):+.4f},{np.max(stamp - rel):+.4f}]"
    )
    for i, name in enumerate(__import__("odom_ml.data.bag", fromlist=["FIELDS"]).FIELDS[topic]):
        col = vals[:, i]
        print(
            f"      {name:9s} min={np.nanmin(col):14.6f} max={np.nanmax(col):14.6f} "
            f"mean={np.nanmean(col):14.6f} nan={int(np.isnan(col).sum())}"
        )

g = grid_from_bag(raw, hz=50.0)
df = g.frame
print(f"\ngrid rows={len(df)} dur={g.duration:.1f}s hz=50")
print(df.describe().T[["min", "max", "mean", "std"]])
print("\nnon-null fraction:")
print(df.notna().mean().round(3).to_string())

vf, vr = df["v_front"].to_numpy(), df["v_rear"].to_numpy()
both = np.isfinite(vf) & np.isfinite(vr)
print(f"\nfront/rear both valid: {both.mean():.3f}  n={both.sum()}")
if both.sum():
    d = vf[both] - vr[both]
    print(f"  diff mean={d.mean():.5f} std={d.std():.5f} p99={np.percentile(np.abs(d), 99):.5f} max={np.abs(d).max():.5f}")

m = np.isfinite(df["mvx"].to_numpy())
if m.sum() > 10:
    gv = np.linalg.norm(df.loc[m, ["mvx", "mvy"]].to_numpy(), axis=1)
    print(f"\ngnss master vel |v| horiz: min={gv.min():.3f} max={gv.max():.3f} mean={gv.mean():.3f}")
    sub = df.loc[m]
    print("  sample rows (t, u, v_front, v_rear, mvx, mvy, mlat, mlon):")
    print(sub[["t", "u", "v_front", "v_rear", "mvx", "mvy", "mlat", "mlon"]].head(12).to_string())
    print("  ... tail:")
    print(sub[["t", "u", "v_front", "v_rear", "mvx", "mvy", "mlat", "mlon"]].tail(8).to_string())
