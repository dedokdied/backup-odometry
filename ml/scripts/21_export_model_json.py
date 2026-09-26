"""Write the slip gate (and the disabled speed head) to flat JSON.

    python ml/scripts/21_export_model_json.py

The export is only useful if a C++ evaluator reproduces sklearn bit for bit, so
the script re-implements the traversal and compares the two on real features
before declaring success.  A mismatch is a hard failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from odom_ml.data.build import HZ, load_labeled  # noqa: E402
from odom_ml.data.features import build_features  # noqa: E402
from odom_ml.models.export import save_json, verify_export  # noqa: E402
from odom_ml.models.speed_slip import load_model  # noqa: E402

TOL = 1e-9


def sample_features(bag_id: str, n: int = 4000) -> np.ndarray:
    d = load_labeled(bag_id, HZ)
    k = min(n, d["t"].size)
    x, _ = build_features(d["u"][:k], d["v_front"][:k], d["v_rear"][:k], hz=HZ)
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/models/speed_slip/speed_slip.joblib")
    ap.add_argument("--out", default="artifacts/models/speed_slip")
    ap.add_argument("--bag", default=None, help="run used for the parity check")
    args = ap.parse_args()

    model = load_model(args.model)
    out = Path(args.out)
    bag = args.bag or "30618_01f73500"
    X = sample_features(bag)
    print(f"parity sample: {X.shape[0]} rows from {bag}")

    meta_common = {
        "source_model": Path(args.model).name,
        "feature_names": model.feature_names,
        "hz": model.hz,
        "parity_bag": bag,
        "parity_rows": int(X.shape[0]),
    }

    results = {}

    slip_path = save_json(
        model.slip_model,
        out / "slip_gate.json",
        model.feature_names,
        kind="classifier",
        metadata={**meta_common, "role": "slip gate", "threshold": model.slip_threshold,
                  "note": "probability that the wheel reading is untrustworthy"},
    )
    doc = json.loads(slip_path.read_text(encoding="utf-8"))
    results["slip_gate"] = verify_export(model.slip_model, doc, X, TOL)

    speed_path = save_json(
        model.speed_model,
        out / "speed_residual.json",
        model.feature_names,
        kind="regressor",
        metadata={**meta_common, "role": "speed residual",
                  "shrink": model.speed_shrink,
                  "note": "DISABLED at inference: validation selected shrink=0, the wheel "
                          "reading is already optimal (adding this makes RMSE worse)"},
    )
    doc = json.loads(speed_path.read_text(encoding="utf-8"))
    results["speed_residual"] = verify_export(model.speed_model, doc, X, TOL)

    for name, r in results.items():
        print(f"\n{name}:")
        print(f"  rows checked   {r['n']}")
        print(f"  max |diff|     {r['max_abs_diff']:.3e}")
        print(f"  mean |diff|    {r['mean_abs_diff']:.3e}")
        print(f"  parity         {'OK' if r['ok'] else 'MISMATCH'}")
        if not r["ok"]:
            raise SystemExit(f"{name} export does not reproduce sklearn")

    for p in (slip_path, speed_path):
        size = p.stat().st_size
        print(f"\nwrote {p}  ({size/1024:.0f} KB)")
        head = p.read_text(encoding="utf-8")[:400]
        print("  first 400 chars: " + head[:400])


if __name__ == "__main__":
    main()
