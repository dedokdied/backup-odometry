from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .. import config as C
from .bag import build_typestore, read_bag
from .grid import grid_from_bag
from .labeling import label_grid

HZ = 50.0
_CACHE = C.CACHE_DIR / "labeled"


def cache_path(bag_id: str, hz: float = HZ) -> Path:
    return _CACHE / f"{bag_id}_{int(hz)}hz.npz"


def build_one(bag_id: str, hz: float = HZ, force: bool = False) -> Path:
    out = cache_path(bag_id, hz)
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    typestore = build_typestore()
    raw = read_bag(bag_id, typestore=typestore)
    grid = grid_from_bag(raw, hz=hz)
    labels = label_grid(grid, hz=hz)
    labels["bag_id"] = np.array(bag_id)
    labels["vehicle"] = np.array(raw.vehicle)
    np.savez_compressed(out, **labels)
    return out


def _worker(args):
    bag_id, hz, force = args
    try:
        p = build_one(bag_id, hz, force)
        return bag_id, str(p), None
    except Exception as exc:  # noqa: BLE001
        return bag_id, None, f"{type(exc).__name__}: {exc}"


def build_many(
    bag_ids: list[str], hz: float = HZ, force: bool = False, workers: int = 6
) -> dict[str, dict]:
    from .bag import list_bags

    bag_ids = bag_ids or list_bags()
    todo = [b for b in bag_ids if force or not cache_path(b, hz).exists()]
    results: dict[str, dict] = {}
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for bag_id, path, err in ex.map(_worker, [(b, hz, force) for b in todo]):
                results[bag_id] = {"path": path, "error": err}
    for b in bag_ids:
        p = cache_path(b, hz)
        if b not in results:
            results[b] = {"path": str(p) if p.exists() else None, "error": None}
    return results


def load_labeled(bag_id: str, hz: float = HZ) -> dict[str, np.ndarray]:
    with np.load(cache_path(bag_id, hz), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_keys(bag_id: str, keys: tuple[str, ...] | list[str], hz: float = HZ) -> dict[str, np.ndarray]:
    """Read only the named arrays out of a cached run.

    The cache holds ~34 arrays of 60 600 float64 samples each and is written with
    ``savez_compressed``, so every array touched costs a zlib inflate.  Callers
    that need three or four columns were paying for all of them: a scan over the
    fleet spent most of its time inflating arrays it never looked at.

    Keys absent from the archive are skipped rather than raising.  The schema is
    bag-dependent -- 39 of the 122 runs carry no GNSS at all, so their cache has
    no ``gnss_*`` entries -- and a caller asking "does this run have a rover fix?"
    needs an answer, not a KeyError.
    """
    with np.load(cache_path(bag_id, hz), allow_pickle=False) as z:
        present = set(z.files)
        return {k: z[k] for k in keys if k in present}


def manifest(hz: float = HZ) -> dict:
    rows = []
    for p in sorted(_CACHE.glob(f"*_{int(hz)}hz.npz")):
        with np.load(p, allow_pickle=False) as z:
            rows.append(
                {
                    "bag_id": str(z["bag_id"]),
                    "vehicle": str(z["vehicle"]),
                    "rows": int(z["t"].size),
                    "duration": float(z["t"][-1]),
                }
            )
    return {"hz": hz, "count": len(rows), "bags": rows}


if __name__ == "__main__":
    res = build_many([], hz=HZ)
    bad = {k: v["error"] for k, v in res.items() if v["error"]}
    print(json.dumps({"ok": len(res) - len(bad), "failed": bad}, ensure_ascii=False, indent=2))
