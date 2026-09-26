"""Run the ported pipeline over one bag and write a features_v2 CSV.

One bag in, one CSV out. The target is exported unclipped with a separate
target_clipped flag; the runtime limit is applied at inference, not here.

Writes exactly the 22 columns the C++ dump has, plus bag_id, so the two can be
compared column by column. The schema is carried in the file name because a dump
without a declared version was silently dropped by the trainer once already.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ml" / "scripts"))
sys.path.insert(0, str(REPO / "ml" / "src"))

from bag_reader import read_bag  # noqa: E402
from odom_ml.features.build_features import (  # noqa: E402
    FEATURE_NAMES,
    build_features,
    build_targets,
)
from odom_ml.features.py_observer import K_B, Observer, adapt_friction  # noqa: E402
from odom_ml.features.py_pipeline import (  # noqa: E402
    M_EFF,
    LowPassFilter,
    accel_label,
    accel_physics,
    adhesion_limit,
    notch_to_u,
    traction,
    wheel_speed_mps,
)
from odom_ml.features.py_slip import MU_PEAK, SlipDetector  # noqa: E402

DT = 0.02
VELOCITY_CUTOFF_HZ = 8.0
WHEEL_RADIUS_M = 0.30


def process_bag(bag_dir: Path, hz: float = 50.0) -> pd.DataFrame:
    df = read_bag(bag_dir, hz=hz)
    n = len(df)
    dt = np.full(n, DT)
    z = np.zeros(n)

    v_front = np.nan_to_num(df["v_front_kmh"].to_numpy()) / 3.6
    v_rear = np.nan_to_num(df["v_rear_kmh"].to_numpy()) / 3.6
    v = wheel_speed_mps(df["v_front_kmh"].to_numpy(), df["v_rear_kmh"].to_numpy())
    a_wheel = np.gradient(LowPassFilter(VELOCITY_CUTOFF_HZ, DT).run(v, DT), DT)

    u = notch_to_u(df["cmd_position"].to_numpy())
    drive, brake, f_net = traction(u, v)
    # Scalar spacing, not the dt array. np.gradient treats a second positional
    # array as coordinates, and an array of equal values has zero spacing.
    cmd_rate = np.abs(np.gradient(u, DT))

    omega_front = v_front / WHEEL_RADIUS_M
    omega_rear = v_rear / WHEEL_RADIUS_M

    sd = SlipDetector()
    obs = Observer()
    mu = MU_PEAK
    cols = np.zeros((n, len(FEATURE_NAMES)))
    mu_track = np.zeros(n)
    a_model_track = np.zeros(n)

    for i in range(n):
        one = slice(i, i + 1)
        f_limit = adhesion_limit(np.array([mu]))[0]
        a_model = accel_physics(v[one], drive[one], brake[one], z[one], z[one], np.array([mu]))[0]
        trust = sd.update(
            DT,
            v_front=v_front[i],
            v_rear=v_rear[i],
            a_wheel=a_wheel[i],
            a_model=a_model,
            drive_force=drive[i],
            brake_force=brake[i],
            adhesion_limit=f_limit,
            effective_mass=M_EFF,
        )
        mu = adapt_friction(
            mu, trust, sd.slip_index, max(drive[i], brake[i]), DT, a_wheel[i],
            adhesion_limit(np.array([mu]))[0],
        )
        f_limit = adhesion_limit(np.array([mu]))[0]
        a_model = accel_physics(v[one], drive[one], brake[one], z[one], z[one], np.array([mu]))[0]
        obs.predict(DT, a_model, v[i], drive[i], brake[i], mu, f_limit)
        obs.update_wheels(omega_front[i], omega_rear[i], trust, sd.slip_index)

        mu_track[i] = mu
        a_model_track[i] = a_model
        cols[i] = build_features(
            u=u[one], v=v[one], a_model=np.array([a_model]),
            omega_front=omega_front[one], omega_rear=omega_rear[one],
            b_scale=np.array([obs.x[K_B]]), slip_index=np.array([sd.slip_index]),
            trust=np.array([trust]), mu=np.array([mu]), dt=dt[one], grade=z[one],
            f_net=np.array([f_net[i]]), f_limit=np.array([f_limit]),
            cmd_rate=cmd_rate[one],
        )[0]

    _, a_lim, _ = accel_label(v)
    a_phys = accel_physics(v, drive, brake, z, z, mu_track)
    target, target_clipped = build_targets(a_lim, a_phys)

    out = pd.DataFrame(cols, columns=list(FEATURE_NAMES))
    out.insert(0, "bag_id", df["bag_id"].iloc[0])
    out.insert(1, "t", df["t"].to_numpy())
    out["target_a_residual"] = target
    out["target_log_scale"] = np.nan
    out["target_mu"] = np.nan
    out["target_clipped"] = target_clipped
    out["wheels_valid"] = 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hz", type=float, default=50.0)
    args = ap.parse_args()

    out = process_bag(args.bag, hz=args.hz)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}  {len(out)} rows, {out.shape[1]} cols")


if __name__ == "__main__":
    main()
