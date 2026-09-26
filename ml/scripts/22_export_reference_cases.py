"""Emit reference cases a C++ port can be checked against.

A C++ node has no access to the bag cache, so the handover ships the raw causal
inputs together with everything the Python side produced from them: the 20
features, the slip probability and the gate decision.  That gives the port two
independent checks -- one on the filters, one on the tree walk -- and either can
fail without the other hiding it.

The builder finishes by reading the JSON back, recomputing from the stored inputs
alone and comparing.  A reference file that does not reproduce is worse than no
file at all, so a mismatch is a hard failure.

The output is written one array per line rather than pretty-printed throughout:
the numeric payload dominates the file size, and keeping each row on a single
line makes it greppable and diffable without bloating it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from odom_ml.data.build import HZ, load_labeled  # noqa: E402
from odom_ml.data.features import (  # noqa: E402
    build_features,
    feature_names,
    implementation_notes,
)
from odom_ml.export import dump_line_json, to_array as _to_array  # noqa: E402
from odom_ml.models.export import predict_json  # noqa: E402
from odom_ml.models.speed_slip import load_model  # noqa: E402

FORMAT = "odom_ml.reference_case/1"
DT = 1.0 / HZ

# float32 round-trips through 9 significant decimal digits
def _num(x) -> str:
    f = float(x)
    return f"{f:.9g}"


def build_case(bag_id: str, rows: int, model, model_doc: dict) -> dict:
    d = load_labeled(bag_id, HZ)
    k = min(rows, d["t"].size)
    u = d["u"][:k].astype(np.float64)
    vf = d["v_front"][:k].astype(np.float64)
    vr = d["v_rear"][:k].astype(np.float64)

    X, spec = build_features(u, vf, vr, hz=HZ)
    p = predict_json(model_doc, X.astype(np.float64))

    return {
        "format": FORMAT,
        "source": {
            "bag_id": bag_id,
            "hz": HZ,
            "rows": k,
            "slice": f"first {k} samples of the run",
            "t_first": float(d["t"][0]),
            "t_last": float(d["t"][k - 1]),
        },
        "feature_spec": spec.as_dict(),
        "implementation_notes": implementation_notes(),
        "tolerance": {
            "features_abs": 1e-6,
            "probability_abs": 1e-9,
            "rationale": (
                "features are float32 on disk, so 1e-6 absolute is at the representation "
                "limit; the probability is float64 math and should match to 1e-12."
            ),
        },
        "inputs": {"u": u, "v_front": vf, "v_rear": vr},
        "expected": {
            "feature_names": feature_names(),
            "features": X,
            "v_wheel": X[:, feature_names().index("v_wheel")].astype(np.float64),
            "slip_probability": p,
            "slip_gate_threshold": float(model.slip_threshold),
            "slip_gate_fired": (p >= model.slip_threshold).astype(np.int64),
        },
    }


def build_case(bag_id: str, rows: int, model, model_doc: dict) -> dict:
    d = load_labeled(bag_id, HZ)
    k = min(rows, d["t"].size)
    u = d["u"][:k].astype(np.float64)
    vf = d["v_front"][:k].astype(np.float64)
    vr = d["v_rear"][:k].astype(np.float64)

    X, spec = build_features(u, vf, vr, hz=HZ)
    p = predict_json(model_doc, X.astype(np.float64))

    return {
        "format": FORMAT,
        "source": {
            "bag_id": bag_id,
            "hz": HZ,
            "rows": k,
            "slice": f"first {k} samples of the run",
            "t_first": float(d["t"][0]),
            "t_last": float(d["t"][k - 1]),
        },
        "feature_spec": spec.as_dict(),
        "implementation_notes": implementation_notes(),
        "tolerance": {
            "features_abs": 1e-6,
            "probability_abs": 1e-9,
            "rationale": (
                "features are float32 on disk, so 1e-6 absolute is at the representation "
                "limit; the probability is float64 math and should match to 1e-12."
            ),
        },
        "inputs": {"u": u, "v_front": vf, "v_rear": vr},
        "expected": {
            "feature_names": feature_names(),
            "features": X,
            "v_wheel": X[:, feature_names().index("v_wheel")].astype(np.float64),
            "slip_probability": p,
            "slip_gate_threshold": float(model.slip_threshold),
            "slip_gate_fired": (p >= model.slip_threshold).astype(np.int64),
        },
    }


def _is_scalar(v) -> bool:
    return not isinstance(v, (dict, list, tuple))


def _fmt_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        # JSON has no NaN literal.  A non-finite input is written as null, which
        # the notes tell the port to read back as NaN -- the raw wheel topics do
        # contain gaps, and the fallback rules for v_wheel have to be exercised.
        if f != f:
            return "null"
        if f in (float("inf"), float("-inf")):
            raise ValueError("infinite value in the reference case; cannot be encoded")
        return _num(f)
    return json.dumps(v, ensure_ascii=False)


def _write(o, indent: int, out: list[str]) -> None:
    """Objects multi-line; arrays of numbers kept on one line each.

    The default pretty-printer turns a 4000x20 matrix into tens of thousands of
    lines, which is unreadable and useless to diff.  This keeps the structure
    visible while a whole feature row stays on one line.
    """
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
            out[-1] += "[" + ", ".join(_fmt_scalar(v) for v in o) + "]"
            return
        if all(isinstance(v, (list, tuple)) for v in o):
            out[-1] += "["
            for i, row in enumerate(o):
                out.append(pad + "  ")
                _write(row, indent + 2, out)
                out[-1] += "," if i < len(o) - 1 else ""
            out.append(pad + "]")
            return
        out[-1] += "["
        for i, v in enumerate(o):
            out.append(pad + "  ")
            _write(v, indent + 2, out)
            out[-1] += "," if i < len(o) - 1 else ""
        out.append(pad + "]")
    else:
        out[-1] += _fmt_scalar(o)


def _round_floats(o):
    """Recursively convert numpy scalars/arrays to plain python."""
    if isinstance(o, dict):
        return {k: _round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round_floats(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return [_round_floats(v) for v in o.tolist()]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def dump_json(case: dict, path: Path) -> None:
    """Compact-but-readable, and valid JSON -- verified by reading it back."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out: list[str] = [""]
    _write(_round_floats(case), 0, out)
    text = "\n".join(x for x in out if x is not None)
    json.loads(text)  # refuse to emit anything the C++ side could not parse
    path.write_text(text + "\n", encoding="utf-8")


def verify(path: Path, model_doc: dict) -> dict[str, float]:
    """Rebuild from the stored inputs alone and compare with the stored outputs."""
    case = json.loads(path.read_text(encoding="utf-8"))
    inp = case["inputs"]
    X, _ = build_features(
        _to_array(inp["u"]), _to_array(inp["v_front"]), _to_array(inp["v_rear"]), hz=HZ
    )
    stored = np.asarray(case["expected"]["features"], dtype=np.float32)
    d_feat = float(np.abs(X - stored).max())
    p = predict_json(model_doc, X.astype(np.float64))
    d_prob = float(np.abs(p - _to_array(case["expected"]["slip_probability"])).max())
    fired = (p >= case["expected"]["slip_gate_threshold"]).astype(int)
    d_gate = int((fired != _to_array(case["expected"]["slip_gate_fired"])).sum())
    tol = case["tolerance"]
    return {
        "rows": int(X.shape[0]),
        "null_inputs": int(
            sum(v is None for k in ("u", "v_front", "v_rear") for v in inp[k])
        ),
        "max_abs_diff_features": d_feat,
        "max_abs_diff_probability": d_prob,
        "gate_disagreements": d_gate,
        "features_ok": d_feat <= tol["features_abs"],
        "probability_ok": d_prob <= tol["probability_abs"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/models/speed_slip/speed_slip.joblib")
    ap.add_argument("--gate", default="artifacts/models/speed_slip/slip_gate.json")
    ap.add_argument("--out", default="artifacts/reference")
    ap.add_argument("--bag", default="30618_01f73500")
    args = ap.parse_args()

    model = load_model(args.model)
    model_doc = json.loads(Path(args.gate).read_text(encoding="utf-8"))
    out = Path(args.out)

    for name, rows in (("reference_case_small.json", 500), ("reference_case_full.json", 4000)):
        case = build_case(args.bag, rows, model, model_doc)
        path = out / name
        dump_line_json(case, path)
        v = verify(path, model_doc)
        print(f"{path}  ({path.stat().st_size/1024:.0f} KB, {v['rows']} rows)")
        print(f"  non-finite inputs (null)  {v['null_inputs']}")
        print(f"  max |diff| features      {v['max_abs_diff_features']:.3e}  "
              f"({'OK' if v['features_ok'] else 'FAIL'}, tol {1e-6:.0e})")
        print(f"  max |diff| probability   {v['max_abs_diff_probability']:.3e}  "
              f"({'OK' if v['probability_ok'] else 'FAIL'}, tol 1e-09)")
        print(f"  gate disagreements       {v['gate_disagreements']}")
        if not (v["features_ok"] and v["probability_ok"] and v["gate_disagreements"] == 0):
            raise SystemExit(f"{name} does not reproduce from its own inputs")
        print("  round-trip from stored inputs: OK")


if __name__ == "__main__":
    main()
