from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .filter.ekf import VelocityEKF
from .models.traction import TractionModel

WHEEL_RATE_MAX = 12.0
DISAGREE_ON = 0.15
DISAGREE_HARD = 0.6
TRUST_DISAGREE = 0.25
TRUST_SLIP = 0.3


def wheel_glitch(value: float, last: float | None, dt: float, rate_max: float = WHEEL_RATE_MAX) -> bool:
    if last is None or dt <= 0:
        return False
    return abs(value - last) > rate_max * max(dt, 1e-3)


@dataclass
class EstimatorConfig:
    dt: float = 0.02
    traction_scale: float = 1.0
    sigma_z: float = 0.05
    q_b: float = 3e-4
    q_k: float = 2e-6
    max_age: float = 0.15
    slip_abs: float = 0.35
    slip_ratio: float = 0.5
    trust_disagree: float = TRUST_DISAGREE
    trust_slip: float = TRUST_SLIP
    use_map: bool = False


@dataclass
class EstimatorState:
    v: float = 0.0
    b: float = 0.0
    k: float = 1.0
    s: float = 0.0
    theta: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    slip: float = 0.0
    trust: float = 1.0
    flags: dict = field(default_factory=dict)


class OdomEstimator:
    """Causal online estimator: inputs (t, u, v_front, v_rear) -> velocity + odometry."""

    def __init__(
        self,
        model: TractionModel | None = None,
        cfg: EstimatorConfig | None = None,
        route=None,
    ) -> None:
        self.model = model
        self.cfg = cfg or EstimatorConfig()
        self.route = route
        self.ekf = VelocityEKF(
            dt=self.cfg.dt,
            sigma_z=self.cfg.sigma_z,
            q_b=self.cfg.q_b,
            q_k=self.cfg.q_k,
        )
        self.hist_f: list[float] = []
        self.hist_r: list[float] = []
        self.last_f: float | None = None
        self.last_r: float | None = None
        self.last_u: float = 0.0
        self.t_f: float | None = None
        self.t_r: float | None = None
        self.t_u: float | None = None
        self.t_last: float | None = None
        self.z_prev: float | None = None
        self.glitch_f = False
        self.glitch_r = False
        self.theta0: float | None = None
        self.s = 0.0
        self.px = 0.0
        self.py = 0.0
        self.pz = 0.0
        self.v_prev = 0.0

    def init_from_gnss(
        self,
        v0: float,
        theta0: float,
        x0: float = 0.0,
        y0: float = 0.0,
        z0: float = 0.0,
    ) -> None:
        self.ekf.x[0] = float(v0)
        self.ekf.P[0, 0] = 0.05**2
        self.theta0 = float(theta0)
        self.s = 0.0
        self.px, self.py, self.pz = x0, y0, z0
        self.v_prev = float(v0)
        if self.route is not None and self.cfg.use_map:
            s0, _ = self.route.project(np.array([[x0, y0]]), window=self.route.length)
            self.s = float(s0[0])

    def _hold(self, value: float | None, t_stamp: float | None, t: float) -> tuple[float, bool]:
        if value is None or t_stamp is None:
            return float("nan"), False
        age = t - t_stamp
        if age < -1e-6 or age > self.cfg.max_age:
            return float(value), False
        return float(value), True

    def _ingest_wheel(
        self,
        value: float | None,
        t: float,
        hist: list[float],
        last: float | None,
        last_t: float | None,
    ) -> tuple[float | None, float | None, bool]:
        if value is None or not np.isfinite(value):
            return last, last_t, False
        if last is not None and last_t is not None and wheel_glitch(value, last, t - last_t):
            return last, last_t, True
        hist.append(float(value))
        if len(hist) > 9:
            hist.pop(0)
        return float(value), t, True

    def step(
        self,
        t: float,
        u: float | None = None,
        v_front: float | None = None,
        v_rear: float | None = None,
    ) -> EstimatorState:
        cfg = self.cfg
        dt = cfg.dt if self.t_last is None else float(np.clip(t - self.t_last, 1e-3, 0.5))
        self.t_last = t

        if u is not None and np.isfinite(u):
            self.last_u = float(u)
            self.t_u = t
        if v_front is not None and np.isfinite(v_front):
            self.last_f, self.t_f, self.glitch_f = self._ingest_wheel(
                v_front, t, self.hist_f, self.last_f, self.t_f
            )
        if v_rear is not None and np.isfinite(v_rear):
            self.last_r, self.t_r, self.glitch_r = self._ingest_wheel(
                v_rear, t, self.hist_r, self.last_r, self.t_r
            )

        uf, ok_f = self._hold(self.last_f, self.t_f, t)
        ur, ok_r = self._hold(self.last_r, self.t_r, t)
        _, ok_u = self._hold(self.last_u, self.t_u, t)

        disagree = ok_f and ok_r and abs(uf - ur) > DISAGREE_ON
        hard = ok_f and ok_r and abs(uf - ur) > DISAGREE_HARD

        if ok_f and ok_r and not hard:
            z, z_ok = 0.5 * (uf + ur), True
        elif ok_f:
            z, z_ok = uf, True
        elif ok_r:
            z, z_ok = ur, True
        else:
            z, z_ok = float("nan"), False

        trust = cfg.trust_disagree if disagree else 1.0
        v_prev = self.ekf.v
        a_tab = float(self.model.accel(self.last_u, max(v_prev, 0.0))) * cfg.traction_scale if self.model else 0.0

        v_wheel_rate = 0.0
        if z_ok and self.z_prev is not None and dt > 0:
            v_wheel_rate = (z - self.z_prev) / dt
        if z_ok:
            self.z_prev = z

        slip = 0.0
        slip_flag = False
        if z_ok and abs(a_tab) > 0.05:
            excess = (v_wheel_rate - a_tab) * np.sign(a_tab)
            slip = float(np.clip(excess / max(abs(a_tab), 0.1), -1.0, 1.0))
            if abs(v_wheel_rate - a_tab) > cfg.slip_abs and slip > cfg.slip_ratio:
                slip_flag = True
                trust = min(trust, cfg.trust_slip)

        self.ekf.step(max(a_tab, 0.0), max(-a_tab, 0.0), z, z_ok, r_scale=1.0 / max(trust, 0.05))
        v = self.ekf.v
        self.s += 0.5 * (v + v_prev) * dt

        st = EstimatorState(
            v=v,
            b=float(self.ekf.x[1]),
            k=float(self.ekf.x[2]),
            s=self.s,
            slip=slip,
            trust=trust,
            flags={
                "front_ok": ok_f,
                "rear_ok": ok_r,
                "u_ok": ok_u,
                "disagree": bool(disagree),
                "hard_disagree": bool(hard),
                "slip": slip_flag,
            },
        )
        self._update_position(st, dt)
        self.v_prev = v
        return st

    def _update_position(self, st: EstimatorState, dt: float) -> None:
        if self.theta0 is None:
            st.x, st.y, st.z = 0.0, 0.0, 0.0
            return
        if self.route is not None and self.cfg.use_map:
            x, y, z, tang = self.route.point_at(np.array([self.s]))
            st.x, st.y, st.z = float(x[0]), float(y[0]), float(z[0])
            st.theta = float(tang[0])
            return
        th = self.theta0
        v_mid = 0.5 * (st.v + self.v_prev)
        self.px += v_mid * dt * float(np.cos(th))
        self.py += v_mid * dt * float(np.sin(th))
        st.x, st.y, st.z = self.px, self.py, self.pz
        st.theta = th
