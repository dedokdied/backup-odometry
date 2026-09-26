from __future__ import annotations

import numpy as np


def vel_metrics(est: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> dict:
    est = np.asarray(est, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    m = np.isfinite(est) & np.isfinite(ref)
    if mask is not None:
        m &= mask
    if m.sum() == 0:
        return {"n": 0}
    e = est[m] - ref[m]
    return {
        "n": int(m.sum()),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mae": float(np.mean(np.abs(e))),
        "bias": float(np.mean(e)),
        "p95": float(np.percentile(np.abs(e), 95)),
        "max": float(np.max(np.abs(e))),
    }


def reg_modes(u: np.ndarray, a: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "accel": u > 0,
        "brake": u < 0,
        "coast": u == 0,
        "stopped": np.abs(a) < 0.05,
    }


def vel_metrics_by_mode(est, ref, u, a) -> dict[str, dict]:
    modes = reg_modes(np.asarray(u), np.asarray(a))
    modes["moving"] = np.asarray(ref) > 1.0
    out = {}
    for name, m in modes.items():
        out[name] = vel_metrics(est, ref, m)
    return out


def along_track_metrics(
    pos_est: np.ndarray,
    pos_ref: np.ndarray,
    path_s: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> dict:
    pos_est = np.asarray(pos_est, dtype=np.float64)
    pos_ref = np.asarray(pos_ref, dtype=np.float64)
    m = np.isfinite(pos_est).all(axis=1) & np.isfinite(pos_ref).all(axis=1)
    if mask is not None:
        m &= mask
    if m.sum() < 2:
        return {"n": 0}
    d = np.linalg.norm(pos_est[m] - pos_ref[m], axis=1)
    traveled = float(np.sum(np.linalg.norm(np.diff(pos_ref[m], axis=0), axis=1)))
    return {
        "n": int(m.sum()),
        "rmse_3d": float(np.sqrt(np.mean(d**2))),
        "mae_3d": float(np.mean(d)),
        "final_error": float(d[-1]),
        "drift_pct": float(100.0 * d[-1] / traveled) if traveled > 1e-6 else float("nan"),
        "traveled": traveled,
    }
