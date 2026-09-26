from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import config as C


@dataclass
class PathGraph:
    name: str
    xy: np.ndarray
    z: np.ndarray
    tang: np.ndarray
    curv: np.ndarray
    s: np.ndarray

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def point_at(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = np.clip(np.asarray(s, dtype=np.float64), 0.0, self.length)
        x = np.interp(s, self.s, self.xy[:, 0])
        y = np.interp(s, self.s, self.xy[:, 1])
        z = np.interp(s, self.s, self.z)
        tang = np.interp(s, self.s, self.tang)
        return x, y, z, tang

    def project(
        self, xy: np.ndarray, s0: float = 0.0, window: float = 60.0
    ) -> tuple[np.ndarray, np.ndarray]:
        best_s = np.full(xy.shape[0], np.nan)
        best_d = np.full(xy.shape[0], np.nan)
        s = s0
        for i in range(xy.shape[0]):
            lo = max(0.0, s - window)
            hi = min(self.length, s + window)
            if hi <= lo:
                lo, hi = 0.0, self.length
            grid = np.arange(lo, hi, 0.25)
            px, py, _, _ = self.point_at(grid)
            d = (px - xy[i, 0]) ** 2 + (py - xy[i, 1]) ** 2
            j = int(np.argmin(d))
            best_s[i] = grid[j]
            best_d[i] = np.sqrt(d[j])
            s = grid[j]
        return best_s, best_d


def load_pathgraph(path: Path | None = None) -> PathGraph:
    path = Path(path) if path is not None else C.PATHGRAPH_DIR / "щукинская - таллинская.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    pts = np.array(
        [[p["x"], p["y"], p["z"], p["tang"], p["curv"]] for p in obj["points"]],
        dtype=np.float64,
    )
    seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return PathGraph(
        name=path.stem,
        xy=pts[:, :2],
        z=pts[:, 2],
        tang=pts[:, 3],
        curv=pts[:, 4],
        s=s,
    )
