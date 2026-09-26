"""Rigid registration of the supplied pathgraph onto the UTM/GNSS frame.

The jury frame is documented as flat MGRS coordinates zeroed at the start of the
recording, so the estimator works in UTM grid axes (easting, northing).  The
supplied pathgraph JSON, however, lives in an unknown local metric frame.  This
module recovers the single rigid transform between the two, once, from the GNSS
tracks, and persists it as an artifact so that inference never has to touch the
map or GNSS at all.

Only a *rigid* transform is fitted: both frames are already metric, so the scale
is reported (and asserted to be ~1) rather than estimated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .. import config as C
from .pathgraph import PathGraph, load_pathgraph

# The pathgraph runs one direction while the tram drives the same corridor back
# and forth, so both traversals are tried and the better one is kept.
FORWARD = "forward"
REVERSE = "reverse"


@dataclass
class RouteRegistration:
    """Rigid transform mapping pathgraph ``(x, y)`` onto UTM ``(E, N)``.

    ``rotation`` is a 2x2 matrix and ``translation`` a 2-vector, so that
    ``utm = rotation @ path_xy.T + translation``.  ``z`` is left untouched:
    the pathgraph heights are absolute altitudes already.
    """

    rotation: np.ndarray
    translation: np.ndarray
    direction: str = FORWARD
    rms_residual_m: float = float("nan")
    median_residual_m: float = float("nan")
    inlier_fraction: float = float("nan")
    scale_estimate: float = 1.0
    iterations: int = 0
    n_correspondences: int = 0
    align_score_m: float = float("nan")
    source_name: str = ""
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def to_local(
        self, easting: np.ndarray, northing: np.ndarray, origin_e: float, origin_n: float
    ) -> np.ndarray:
        """UTM metres relative to the recording origin, as ``(x, y)``."""
        d = np.column_stack(
            [np.asarray(easting, dtype=np.float64), np.asarray(northing, dtype=np.float64)]
        )
        return d - np.array([origin_e, origin_n], dtype=np.float64)

    def to_utm(self, xy: np.ndarray) -> np.ndarray:
        return np.asarray(xy, dtype=np.float64) @ self.rotation.T + self.translation

    def to_pathgraph(self, easting: np.ndarray, northing: np.ndarray) -> np.ndarray:
        inv = self.rotation.T
        d = np.column_stack(
            [np.asarray(easting, dtype=np.float64), np.asarray(northing, dtype=np.float64)]
        ) - self.translation
        return d @ inv.T

    def to_dict(self) -> dict:
        return {
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "direction": self.direction,
            "rms_residual_m": self.rms_residual_m,
            "median_residual_m": self.median_residual_m,
            "inlier_fraction": self.inlier_fraction,
            "scale_estimate": self.scale_estimate,
            "iterations": self.iterations,
            "n_correspondences": self.n_correspondences,
            "align_score_m": self.align_score_m,
            "source_name": self.source_name,
            "notes": self.notes,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, obj: dict) -> RouteRegistration:
        known = {
            "rotation",
            "translation",
            "direction",
            "rms_residual_m",
            "median_residual_m",
            "inlier_fraction",
            "scale_estimate",
            "iterations",
            "n_correspondences",
            "align_score_m",
            "source_name",
            "notes",
        }
        return cls(
            rotation=np.asarray(obj["rotation"], dtype=np.float64),
            translation=np.asarray(obj["translation"], dtype=np.float64),
            direction=obj.get("direction", FORWARD),
            rms_residual_m=obj.get("rms_residual_m", float("nan")),
            median_residual_m=obj.get("median_residual_m", float("nan")),
            inlier_fraction=obj.get("inlier_fraction", float("nan")),
            scale_estimate=obj.get("scale_estimate", 1.0),
            iterations=obj.get("iterations", 0),
            n_correspondences=obj.get("n_correspondences", 0),
            align_score_m=obj.get("align_score_m", float("nan")),
            source_name=obj.get("source_name", ""),
            notes=obj.get("notes", ""),
            extra={k: v for k, v in obj.items() if k not in known},
        )


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares similarity fit ``dst ~ scale * src @ rot.T + t``.

    ``src``/``dst`` are row-major ``(n, 2)`` arrays.  The SVD is taken on the
    cross-covariance ``cov = mean(dst s^T)``; because points are stored as rows,
    the returned ``rot`` is the *transpose* of the usual column-vector solution,
    which is what makes the ``src @ rot.T`` application below correct.
    """
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    s0 = src - mu_s
    d0 = dst - mu_d
    cov = d0.T @ s0 / src.shape[0]
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        sign[1, 1] = -1.0
    rot = u @ sign @ vt
    var = float((s0**2).sum() / src.shape[0])
    scale = float((np.diag(d) * np.diag(sign)).sum() / var) if var > 0 else 1.0
    trans = mu_d - scale * (src @ rot.T).mean(axis=0)
    return rot, trans, scale


def _rigid(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rot, trans, _ = _umeyama(src, dst)
    return rot, trans


def _trimmed_residuals(
    src: np.ndarray, target_tree: cKDTree, rot: np.ndarray, trans: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    moved = src @ rot.T + trans
    dist, idx = target_tree.query(moved, k=1)
    return dist, idx


def _align_score(moved: np.ndarray, tree: cKDTree) -> float:
    """Median nearest-neighbour distance of the moved cloud onto the target.

    The median (not a trimmed RMS) is deliberate: a trimmed score happily accepts
    an alignment that matches a *minority* of the route well and leaves the rest
    hundreds of metres away, which is exactly the mirrored-basin failure mode of
    a symmetric out-and-back corridor.  The median breaks at 50 % outliers, so
    sparse GNSS dropouts still do not dominate.
    """
    dist, _ = tree.query(moved, k=1)
    return float(np.median(dist))


def _coarse_rotation(
    src_c: np.ndarray,
    mu_t: np.ndarray,
    tree: cKDTree,
    steps_deg=(5.0, 1.0, 0.25),
    top_k: int = 6,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Rank rotations by a median-offset alignment score, best first.

    ``src_c`` must already be median-centred, and the returned translation is
    expressed for that centred cloud, so it can be handed straight to
    :func:`_refine`.  The route is a thin out-and-back corridor whose two clouds
    have different centroids and different coverage, so aligning centroids gives
    a bad initial translation that ICP cannot escape; re-solving the offset for
    every candidate rotation keeps the sweep meaningful.  Several candidates are
    returned because a nearly symmetric corridor scores the same under rotations
    180 deg apart, and only the ICP stage can tell those apart.
    """
    scored: list[tuple[float, float]] = []
    for deg in np.arange(0.0, 360.0, steps_deg[0]):
        a = np.deg2rad(deg)
        rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        # The moved cloud's centroid is exactly ``trans``, so matching medians
        # means ``trans = mu_t``.
        scored.append((_align_score(src_c @ rot.T + mu_t, tree), float(deg)))
    for step in steps_deg[1:]:
        for deg in np.arange(0.0, 360.0, step):
            a = np.deg2rad(deg)
            rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
            scored.append((_align_score(src_c @ rot.T + mu_t, tree), float(deg)))
    scored.sort()

    picked: list[tuple[float, float]] = []
    for score, deg in scored:
        if all(min(abs(deg - d), 360.0 - abs(deg - d)) > 15.0 for _, d in picked):
            picked.append((score, deg))
        if len(picked) >= top_k:
            break

    out = []
    for score, deg in picked:
        a = np.deg2rad(deg)
        rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        out.append((rot, mu_t, score))
    return out


def _refine(
    src_c: np.ndarray,
    tgt: np.ndarray,
    tree: cKDTree,
    rot: np.ndarray,
    trans: np.ndarray,
    trim: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Monotone trimmed ICP from one initial guess, on a median-centred cloud."""
    n_keep = max(3, int(round(src_c.shape[0] * trim)))
    best_rot, best_trans = rot, trans
    best_score = _align_score(src_c @ rot.T + trans, tree)
    iters = 0
    for iters in range(1, max_iter + 1):
        dist, idx = _trimmed_residuals(src_c, tree, best_rot, best_trans)
        part = np.argpartition(dist, n_keep - 1)[:n_keep]
        rot, trans = _rigid(src_c[part], tgt[idx[part]])
        score = _align_score(src_c @ rot.T + trans, tree)
        if score >= best_score - tol * max(1.0, best_score):
            break
        best_rot, best_trans, best_score = rot, trans, score
    return best_rot, best_trans, best_score, iters


def fit_registration(
    path_xy: np.ndarray,
    target_xy: np.ndarray,
    trim: float = 0.8,
    max_iter: int = 80,
    tol: float = 1e-4,
    subsample: int = 1500,
    seed: int = 0,
) -> RouteRegistration:
    """Fit a rigid transform from ``path_xy`` onto ``target_xy`` with trimmed ICP."""
    path_xy = np.asarray(path_xy, dtype=np.float64)[:, :2]
    target_xy = np.asarray(target_xy, dtype=np.float64)[:, :2]
    if path_xy.shape[0] < 3 or target_xy.shape[0] < 3:
        raise ValueError("need at least 3 points on each side to register")

    rng = np.random.default_rng(seed)
    if path_xy.shape[0] > subsample:
        src = path_xy[rng.choice(path_xy.shape[0], subsample, replace=False)]
    else:
        src = path_xy
    if target_xy.shape[0] > max(20000, 8 * src.shape[0]):
        step = int(np.ceil(target_xy.shape[0] / max(20000, 8 * src.shape[0])))
        tgt = target_xy[::step]
    else:
        tgt = target_xy

    tree = cKDTree(tgt)
    # Work on a median-centred source so that the coarse sweep's translation and
    # the ICP correspondences share one frame; the offset is folded back into the
    # reported translation at the end.
    mu_s = np.median(src, axis=0)
    src_c = src - mu_s
    mu_t = np.median(tgt, axis=0)

    candidates = _coarse_rotation(src_c, mu_t, tree)
    best: tuple[np.ndarray, np.ndarray, float, int] | None = None
    for rot0, trans0, _ in candidates:
        out = _refine(src_c, tgt, tree, rot0, trans0, trim, max_iter, tol)
        if best is None or out[2] < best[2]:
            best = out
    assert best is not None
    rot, trans_c, best_score, iters = best
    # src_c @ rot.T + trans_c  ==  src @ rot.T + (trans_c - mu_s @ rot.T)
    trans = trans_c - mu_s @ rot.T

    dist, idx = _trimmed_residuals(src, tree, rot, trans)
    n_keep = max(3, int(round(src.shape[0] * trim)))
    part = np.argpartition(dist, n_keep - 1)[:n_keep]
    kept = dist[part]
    _, _, scale = _umeyama(src[part], tgt[idx[part]])
    return RouteRegistration(
        rotation=rot,
        translation=trans,
        rms_residual_m=float(np.sqrt((kept**2).mean())),
        median_residual_m=float(np.median(dist)),
        inlier_fraction=float((dist < 5.0).mean()),
        scale_estimate=scale,
        iterations=iters,
        n_correspondences=int(src.shape[0]),
        align_score_m=best_score,
    )


def register_pathgraph(
    path_xy: np.ndarray,
    target_xy: np.ndarray,
    trim: float = 0.8,
    **kwargs,
) -> RouteRegistration:
    """Fit both traversals of an out-and-back pathgraph and keep the better fit.

    The corridor is symmetric, so a fit can lock onto the mirrored leg.  Scoring
    both the forward and the reversed correspondence and keeping the lower
    residual is what makes the result stable.
    """
    best: RouteRegistration | None = None
    for direction, src in ((FORWARD, path_xy), (REVERSE, path_xy[::-1])):
        reg = fit_registration(src, target_xy, trim=trim, **kwargs)
        reg.direction = direction
        if best is None or reg.rms_residual_m < best.rms_residual_m:
            best = reg
    assert best is not None
    return best


def registration_path() -> Path:
    return C.ARTIFACTS_DIR / "route_registration.json"


def save_registration(reg: RouteRegistration, path: Path | None = None) -> Path:
    path = registration_path() if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_registration(path: Path | None = None) -> RouteRegistration:
    path = registration_path() if path is None else Path(path)
    return RouteRegistration.from_dict(json.loads(path.read_text(encoding="utf-8")))


def build_registration(
    graph: PathGraph | None = None,
    target_xy: np.ndarray | None = None,
    trim: float = 0.8,
    **kwargs,
) -> RouteRegistration:
    """Register ``graph`` (default: the Shchukin-Tallinn pathgraph) onto UTM points."""
    graph = load_pathgraph() if graph is None else graph
    if target_xy is None:
        raise ValueError("target_xy (UTM easting/northing of the GNSS tracks) is required")
    reg = register_pathgraph(graph.xy, target_xy, trim=trim, **kwargs)
    reg.source_name = graph.name
    return reg
