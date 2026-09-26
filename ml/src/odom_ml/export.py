"""Line-oriented JSON writer for the handover artefacts.

The default pretty-printer explodes a 4000x20 matrix into tens of thousands of
lines, which no one can read or diff.  This keeps objects multi-line but writes
each row of numbers on a single line, so the structure stays visible and a feature
row stays greppable.

Non-finite inputs are written as ``null``: JSON has no NaN literal, and the raw
wheel topics do contain gaps that the feature fallback rules have to be tested
against.  Reading the artefacts back must turn ``null`` into NaN, which
:func:`to_array` does.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# float32 round-trips through 9 significant decimal digits
FLOAT_FMT = "{:.9g}"


def _num(x) -> str:
    return FLOAT_FMT.format(float(x))


def _is_scalar(v) -> bool:
    return not isinstance(v, (dict, list, tuple))


def fmt_scalar(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "true" if bool(v) else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if f != f:
            return "null"  # NaN -> null, read back with to_array()
        if f in (float("inf"), float("-inf")):
            raise ValueError("infinite value cannot be encoded in JSON")
        return _num(f)
    return json.dumps(v, ensure_ascii=False)


def _write(o, indent: int, out: list[str]) -> None:
    pad = " " * indent
    if isinstance(o, dict):
        if not o:
            out[-1] += "{}"
            return
        out[-1] += "{"
        for i, (k, v) in enumerate(o.items()):
            out.append(f"{pad}  {json.dumps(str(k), ensure_ascii=False)}: ")
            _write(v, indent + 2, out)
            out[-1] += "," if i < len(o) - 1 else ""
        out.append(pad + "}")
    elif isinstance(o, (list, tuple)):
        if len(o) == 0:
            out[-1] += "[]"
            return
        if all(_is_scalar(v) for v in o):
            out[-1] += "[" + ", ".join(fmt_scalar(v) for v in o) + "]"
            return
        out[-1] += "["
        for i, v in enumerate(o):
            out.append(pad + "  ")
            _write(v, indent + 2, out)
            out[-1] += "," if i < len(o) - 1 else ""
        out.append(pad + "]")
    else:
        out[-1] += fmt_scalar(o)


def plain(o):
    """Recursively turn numpy scalars and arrays into plain python."""
    if isinstance(o, dict):
        return {k: plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return [plain(v) for v in o.tolist()]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def dump_line_json(obj, path: Path) -> Path:
    """Write the document and refuse to emit anything that will not parse back."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out: list[str] = [""]
    _write(plain(obj), 0, out)
    text = "\n".join(out)
    json.loads(text)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def to_array(values) -> np.ndarray:
    """JSON values back to float64, with ``null`` restored to NaN."""
    return np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)
