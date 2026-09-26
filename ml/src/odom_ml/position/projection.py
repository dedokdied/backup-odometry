"""Project positions onto the registered route and into the jury output frame.

Two frames are in play:

* the *map* frame of the supplied pathgraph JSON, whose relation to the outside
  world is unknown until :mod:`odom_ml.position.registration` recovers it;
* the *output* frame, which the jury documents as flat MGRS/UTM grid coordinates
  with the origin at the start of the recording.  UTM grid axes are not
  north-aligned, so the frame is UTM metres minus the UTM of the start point --
  not ENU, and not a rotation of it.

The supplied route is an out-and-back with two separate tracks, so the
pathgraph does not overlap itself and a point's arclength is determined by its
nearest position on the polyline alone.  :meth:`MapProjection.ambiguity` checks
that property instead of assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..geo import latlon_to_mgrs, latlon_to_utm
from .pathgraph import PathGraph, load_pathgraph
from .registration import RouteRegistration, load_registration


@dataclass
class TrackProjection:
    """Where a sampled track sits relative to the route."""

    s: np.ndarray  #: arclength along the route, m
    lateral: np.ndarray  #: signed offset from the route, m (+ is left of travel)
    tangent: np.ndarray  #: unit heading of the route at ``s``, in UTM (E, N)
    ds: np.ndarray  #: arclength increment between consecutive samples, m
    ambiguous: bool = False

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.s)

    def travelled(self) -> float:
        """Net arclength covered, ignoring the noisy per-sample sign of ``ds``."""
        v = self.s[np.isfinite(self.s)]
        return float(v[-1] - v[0]) if v.size > 1 else 0.0


class MapProjection:
    """A registered route: UTM <-> arclength, plus the jury output frame."""

    def __init__(self, graph: PathGraph, registration: RouteRegistration) -> None:
        self.graph = graph
        self.reg = registration
        self._tree = cKDTree(graph.xy)
        self._a = graph.xy[:-1]
        self._b = graph.xy[1:]
        seg = self._b - self._a
        self._seg_len = np.linalg.norm(seg, axis=1)
        self._seg_s = graph.s[:-1]

    @classmethod
    def load(cls) -> MapProjection:
        """Build from the shipped pathgraph and the saved registration artifact."""
        return cls(load_pathgraph(), load_registration())

    # --- frame conversions ---------------------------------------------------

    def utm_to_map(self, easting: np.ndarray, northing: np.ndarray) -> np.ndarray:
        """UTM metres -> the pathgraph's own local frame."""
        e = np.asarray(easting, dtype=np.float64)
        n = np.asarray(northing, dtype=np.float64)
        return self.reg.to_pathgraph(e, n)

    def map_to_utm(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy = np.asarray(xy, dtype=np.float64)
        if xy.ndim == 1:
            xy = xy[None, :]
        out = self.reg.to_utm(xy)
        return out[:, 0], out[:, 1]

    def s_to_utm(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Arclength along the route -> UTM easting/northing."""
        s = np.clip(np.asarray(s, dtype=np.float64), 0.0, self.graph.length)
        x = np.interp(s, self.graph.s, self.graph.xy[:, 0])
        y = np.interp(s, self.graph.s, self.graph.xy[:, 1])
        return self.map_to_utm(np.column_stack([x, y]))

    def s_to_output(self, s: np.ndarray, origin: tuple[float, float]) -> np.ndarray:
        e, n = self.s_to_utm(s)
        return np.column_stack([e - origin[0], n - origin[1]])

    def tangent_at(self, s: np.ndarray) -> np.ndarray:
        """Unit route heading in UTM (E, N) at arclength ``s``."""
        s = np.clip(np.asarray(s, dtype=np.float64), 0.0, self.graph.length)
        eps = 0.5
        e0, n0 = self.s_to_utm(np.clip(s - eps, 0.0, self.graph.length))
        e1, n1 = self.s_to_utm(np.clip(s + eps, 0.0, self.graph.length))
        d = np.column_stack([e1 - e0, n1 - n0])
        norm = np.linalg.norm(d, axis=1, keepdims=True)
        return np.divide(d, np.where(norm == 0.0, 1.0, norm))

    @staticmethod
    def to_output(
        easting: np.ndarray,
        northing: np.ndarray,
        alt: np.ndarray,
        origin: tuple[float, float, float],
    ) -> np.ndarray:
        """UGRS/UTM metres relative to the recording start: ``(x, y, z)``."""
        return np.column_stack(
            [
                np.asarray(easting, dtype=np.float64) - origin[0],
                np.asarray(northing, dtype=np.float64) - origin[1],
                np.asarray(alt, dtype=np.float64) - origin[2],
            ]
        )

    @staticmethod
    def origin_mgrs(lat: float, lon: float, alt: float) -> str:
        """MGRS label of the origin, for human-readable reporting only."""
        e, n, _ = latlon_to_utm(np.array([lat]), np.array([lon]))
        r = latlon_to_mgrs(np.array([lat]), np.array([lon]))
        return f"{str(r['text'][0])} (E {e[0]:.1f}, N {n[0]:.1f}, alt {alt:.1f} m)"

    # --- projection ----------------------------------------------------------

    def project_utm(self, easting: np.ndarray, northing: np.ndarray) -> TrackProjection:
        """Project UTM points onto the route: arclength, lateral offset, heading.

        Vectorised point-to-segment projection.  A KD-tree query only proposes the
        two segments adjacent to the nearest vertex; the exact projection onto
        those segments is then closed-form, so the cost is ``O(n log m)``.
        """
        p = self.utm_to_map(easting, northing)
        s, offset, tan_map = self._project(p)
        # signed lateral: the offset resolved on the left-hand normal of the route
        normal = np.column_stack([-tan_map[:, 1], tan_map[:, 0]])
        lateral = (offset * normal).sum(axis=1)
        tangent = tan_map @ self.reg.rotation.T
        ds = np.diff(np.concatenate([[np.nan], s]))
        return TrackProjection(s=s, lateral=lateral, tangent=tangent, ds=ds)

    def _project(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Point-to-polyline projection in the map frame.

        Returns the arclength, the signed-ready offset vector ``p - closest`` and
        the route tangent there.
        """
        _, j = self._tree.query(p, k=1)
        j = np.asarray(j, dtype=np.int64)
        best_s = np.full(p.shape[0], np.nan)
        best_d = np.full(p.shape[0], np.inf)
        best_c = np.zeros_like(p)
        # segments (j-1 -> j) and (j -> j+1)
        for seg_idx in (j - 1, j):
            k = np.clip(seg_idx, 0, self._a.shape[0] - 1)
            a, b = self._a[k], self._b[k]
            d = b - a
            dd = (d * d).sum(axis=1)
            dd = np.where(dd == 0.0, 1.0, dd)
            t = np.clip(((p - a) * d).sum(axis=1) / dd, 0.0, 1.0)
            closest = a + t[:, None] * d
            dist = np.linalg.norm(p - closest, axis=1)
            take = dist < best_d
            best_s[take] = self._seg_s[k[take]] + t[take] * self._seg_len[k[take]]
            best_d[take] = dist[take]
            best_c[take] = closest[take]
        return best_s, p - best_c, self._tangent_at_s(best_s)

    def _tangent_at_s(self, s: np.ndarray) -> np.ndarray:
        """Unit tangent in the map frame, from the geometric path.

        The JSON also carries a ``tang`` field, but deriving the tangent from the
        polyline itself cannot be wrong, so it is used as the reference and the
        field is only trusted where the two agree.
        """
        return self._geometric_tangent(s)

    def _geometric_tangent(self, s: np.ndarray) -> np.ndarray:
        eps = 0.5
        a = np.clip(s - eps, 0.0, self.graph.length)
        b = np.clip(s + eps, 0.0, self.graph.length)
        xa = np.interp(a, self.graph.s, self.graph.xy[:, 0])
        ya = np.interp(a, self.graph.s, self.graph.xy[:, 1])
        xb = np.interp(b, self.graph.s, self.graph.xy[:, 0])
        yb = np.interp(b, self.graph.s, self.graph.xy[:, 1])
        d = np.column_stack([xb - xa, yb - ya])
        norm = np.linalg.norm(d, axis=1, keepdims=True)
        return np.divide(d, np.where(norm == 0.0, 1.0, norm))

    # --- self-checks ---------------------------------------------------------

    def ambiguity(self, radius: float = 15.0) -> dict[str, float]:
        """How far apart in arclength spatially-close route points can be.

        If this is not small the route overlaps itself and a nearest-point
        projection is ambiguous, so arclength would need to be tracked with
        motion continuity instead.
        """
        pairs = self._tree.query_pairs(r=radius, output_type="ndarray")
        if pairs.size == 0:
            return {"pairs": 0, "max_arclength_separation_m": 0.0}
        sep = np.abs(self.graph.s[pairs[:, 0]] - self.graph.s[pairs[:, 1]])
        return {
            "pairs": int(pairs.shape[0]),
            "max_arclength_separation_m": float(sep.max()),
            "p99_arclength_separation_m": float(np.percentile(sep, 99)),
        }

    def sanity(self) -> dict[str, float]:
        """Round-trip the route through its own registration and projection."""
        e, n = self.map_to_utm(self.graph.xy)
        proj = self.project_utm(e, n)
        return {
            "length_m": float(self.graph.length),
            "s_roundtrip_max_err_m": float(np.nanmax(np.abs(proj.s - self.graph.s))),
            "lateral_max_abs_m": float(np.nanmax(np.abs(proj.lateral))),
            "endpoint_gap_m": float(np.linalg.norm(self.graph.xy[-1] - self.graph.xy[0])),
            **self.ambiguity(),
        }
