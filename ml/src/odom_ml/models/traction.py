from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

U_MIN, U_MAX = -15, 15
V_MAX = 16.0
DV = 0.5
NU = U_MAX - U_MIN + 1
NV = int(V_MAX / DV) + 1
V_EDGES = np.arange(NV + 1) * DV


def u_index(u: np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(u, dtype=np.float64) - U_MIN, 0.0, NU - 1.0)


def v_index(v: np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(v, dtype=np.float64) / DV, 0.0, NV - 1.0 - 1e-9)


def table_lookup(table: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    fu = u_index(u)
    fv = v_index(v)
    i0 = np.floor(fu).astype(np.int64)
    j0 = np.floor(fv).astype(np.int64)
    i0 = np.clip(i0, 0, NU - 2)
    j0 = np.clip(j0, 0, NV - 2)
    du = fu - i0
    dv = fv - j0
    t = table
    c00 = t[i0, j0]
    c10 = t[i0 + 1, j0]
    c01 = t[i0, j0 + 1]
    c11 = t[i0 + 1, j0 + 1]
    return (c00 * (1 - du) * (1 - dv) + c10 * du * (1 - dv) + c01 * (1 - du) * dv + c11 * du * dv)


@dataclass
class TractionModel:
    table: np.ndarray
    meta: dict = field(default_factory=dict)

    def accel(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return table_lookup(self.table, u, v)

    def to_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, table=self.table.astype(np.float32))
        (path.parent / "traction_meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def from_npz(path: Path) -> "TractionModel":
        with np.load(path) as z:
            table = z["table"].astype(np.float64)
        meta_path = Path(path).parent / "traction_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return TractionModel(table=table, meta=meta)


def smooth_table(table: np.ndarray, wu: int = 3, wv: int = 3) -> np.ndarray:
    out = table.astype(np.float64).copy()
    for _ in range(2):
        pad_u = np.pad(out, ((wu, wu), (0, 0)), mode="edge")
        out = sum(pad_u[i : i + out.shape[0]] for i in range(2 * wu + 1)) / (2 * wu + 1)
        pad_v = np.pad(out, ((0, 0), (wv, wv)), mode="edge")
        out = sum(pad_v[:, i : i + out.shape[1]] for i in range(2 * wv + 1)) / (2 * wv + 1)
    return out


def enforce_traction_monotonicity(table: np.ndarray) -> np.ndarray:
    out = table.copy()
    traction_rows = np.arange(1, NU)
    for j in range(out.shape[1]):
        col = out[traction_rows, j]
        out[traction_rows, j] = np.maximum.accumulate(col)
    return out
