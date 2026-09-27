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


# GNSS may be used for initial alignment only. The judging rule is a 2.5 s
# window; everything after it is dropped from the dump, and a bag that carries
# no GNSS at all gets NaN throughout.
GNSS_EXPORT_WINDOW_S = 2.5

#: exported name -> name bag_reader produces
GNSS_EXPORT_COLUMNS = (
    ("gnss_master_lat", "master_fix_lat"),
    ("gnss_master_lon", "master_fix_lon"),
    ("gnss_master_alt", "master_fix_alt"),
    ("gnss_rover_lat", "rover_fix_lat"),
    ("gnss_rover_lon", "rover_fix_lon"),
    ("gnss_rover_alt", "rover_fix_alt"),
    ("gnss_master_vel_x", "master_vel_x"),
    ("gnss_master_vel_y", "master_vel_y"),
    ("gnss_rover_vel_x", "rover_vel_x"),
    ("gnss_rover_vel_y", "rover_vel_y"),
)

GNSS_MAX_AGE_S = 0.35
GNSS_MIN_SATS = 6
GNSS_INIT_WINDOW_S = 2.5
MIN_TRAVEL_FOR_CALIB_M = 30.0
CAL_WINDOW_SAMPLES = 250  # 5 s at 50 Hz
CAL_MAX_ABS_A = 0.2  # steady state only, the same gate adapt_friction uses  # 5 s at 50 Hz


def _gnss_state(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(v_gnss horizontal m/s, valid mask), following estimator.cpp:239-241.

    NavSatFix carries no satellite count, so the C++ gate

        if (g.sats > 0 && g.sats < gnss_min_sats) return false;

    cannot be reproduced. It rejects only a *known* low count and accepts an
    unknown one, so the faithful port here accepts too. position_covariance_type
    is not a usable stand-in: this dataset leaves it at 0
    (COVARIANCE_UNKNOWN) on every sample, so gating on it rejects all 73 023
    rows of a bag that has perfectly good GNSS. Satellites are simply not
    available from this message, and the observer's own chi-square gate plus the
    absolute innovation bound are what actually reject bad fixes.
    """
    vx = np.nan_to_num(df["master_vel_x"].to_numpy())
    vy = np.nan_to_num(df["master_vel_y"].to_numpy())
    v_gnss = np.hypot(vx, vy)
    has_vel = np.isfinite(df["master_vel_x"].to_numpy()) | np.isfinite(
        df["master_vel_y"].to_numpy()
    )
    return v_gnss, has_vel


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
    gnss_used = np.zeros(n, dtype=np.int8)
    b_scale_track = np.zeros(n)

    v_gnss, gnss_valid = _gnss_state(df)
    t = df["t"].to_numpy()
    # First index at which the GNSS reference exists; the C++ window is measured
    # from init_t0_, which is the first accepted fix.
    gnss_idx = np.flatnonzero(gnss_valid)
    init_t0 = t[gnss_idx[0]] if gnss_idx.size else None
    v_wheel_mean = 0.5 * (v_front + v_rear)
    travelled = 0.0
    v_prev = 0.0
    scale_cal_done = False
    cal_samples: list = []
    cal_result = None

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

        # --- 6. GNSS updates, estimator.cpp:576-596.
        # The scale calibration is ONE-SHOT on accumulated travel, not the
        # 2.5 s init window. The tram starts from a standstill, so inside that
        # window |v_wheel| is exactly 0 and calibrate_scale_from_velocity returns
        # immediately; by the time the tram moves the window has closed. That
        # leaves b_scale pinned at 1.0 for every run, which is the systematic
        # wheel-scale error showing up as integrated position drift.
        #
        # One shot, not continuous. The measurement form self-damps for a single
        # innovation, but a continuous 50 Hz stream of them integrates GNSS noise
        # into b until P(kB,kB) collapses, and b then wanders into the 0.80/1.25
        # clamps: measured std 0.027-0.038, three bags saturating. The ratio is
        # taken as a median over a window instead, which rejects the outliers
        # that a single instantaneous sample would hand straight to the filter.
        travelled += abs(v[i] - v_prev) if i else 0.0
        v_prev = v[i]
        if init_t0 is not None and gnss_valid[i]:
            obs.update_gnss_velocity(v_gnss[i])
        if (
            not scale_cal_done
            and init_t0 is not None
            and gnss_valid[i]
            and travelled >= MIN_TRAVEL_FOR_CALIB_M
            and abs(v_wheel_mean[i]) >= 1.0
            and abs(a_wheel[i]) <= CAL_MAX_ABS_A
        ):
            cal_samples.append((v_gnss[i], v_wheel_mean[i]))
            if len(cal_samples) >= CAL_WINDOW_SAMPLES:
                pairs = np.asarray(cal_samples)
                v_ref = float(np.median(pairs[:, 0]))
                v_whl = float(np.median(pairs[:, 1]))
                obs.calibrate_scale_from_velocity(v_ref, v_whl)
                scale_cal_done = True
                cal_result = (v_ref, v_whl, v_ref / v_whl)

        mu_track[i] = mu
        b_scale_track[i] = obs.x[K_B]
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
    out["gnss_used"] = gnss_used
    out["scale_cal_v_ref"] = 0.0 if cal_result is None else cal_result[0]
    out["scale_cal_v_wheel"] = 0.0 if cal_result is None else cal_result[1]
    out["scale_cal_ratio"] = 0.0 if cal_result is None else cal_result[2]

    # GNSS export, restricted to the alignment window. The features themselves
    # never read these columns; they exist so the C++ harness can drive
    # Estimator::step the way the runtime does, not so the model can see GNSS.
    t_first = float(df["t"].iloc[0])
    within = (df["t"].to_numpy() - t_first) < GNSS_EXPORT_WINDOW_S
    for col, src in GNSS_EXPORT_COLUMNS:
        if src in df.columns:
            out[col] = np.where(within, df[src].to_numpy(dtype=np.float64), np.nan)
        else:
            out[col] = np.full(len(df), np.nan)
    out["b_scale_final"] = b_scale_track
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
