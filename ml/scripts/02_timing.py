import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from odom_ml import config as C
from odom_ml.data import build_typestore, read_bag

bags = sys.argv[1:] or ["30618_0e41eac3", "30639_e4379d7f", "30618_27e994fc"]
ts = build_typestore()

for bag_id in bags:
    raw = read_bag(bag_id, typestore=ts)
    print(f"\n=== {bag_id}  vehicle={raw.vehicle}  dur={raw.duration:.1f}s")
    for topic in C.ALL_TOPICS:
        t = raw.t[topic].astype(np.float64) - raw.t0_ns
        if t.size < 2:
            print(f"  {topic:34s} n={t.size}")
            continue
        dt = np.diff(t) / 1e9
        stamp = (raw.header_stamp[topic] - raw.t0_ns) / 1e9
        off = stamp - t / 1e9
        print(
            f"  {topic:34s} n={t.size:7d} dt[med={np.median(dt):.4f} "
            f"p99={np.percentile(dt, 99):.4f} max={dt.max():.3f}] "
            f"stamp-t0=[{off.min():+.3f},{off.max():+.3f}]"
        )
