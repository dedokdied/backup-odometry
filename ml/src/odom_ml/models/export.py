"""Export a fitted HistGradientBoosting model to flat JSON, and read it back.

Why JSON rather than ONNX: the runtime artefact here is a few hundred kilobytes
of thresholds that any C++ node can evaluate in thirty lines, with no onnxruntime
dependency -- which matters because the case has to build with ``colcon`` offline
on two cores.  ONNX can still be produced from the same intermediate
representation once the integration is working.

The traversal rule mirrors ``sklearn``'s own evaluator exactly:

* a NaN feature follows ``missing_go_to_left``;
* otherwise ``value <= num_threshold`` goes left, ``value > num_threshold`` right;
* a leaf contributes its ``value``.

Nothing here is trusted on faith: :func:`predict_json` re-implements the rule and
``verify_export`` compares it against ``sklearn``'s own output, so a wrong
threshold, a wrong child index or a missing baseline offset shows up as a
mismatch instead of a silent difference in the C++ node.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

JSON_FORMAT = "odom_ml.hgb_json/1"


def _tree_nodes(predictor) -> list[dict]:
    """Flatten one sklearn TreePredictor into index-addressable JSON nodes."""
    n = predictor.nodes
    out: list[dict] = []
    for i in range(n.shape[0]):
        if n["is_leaf"][i]:
            out.append({"is_leaf": True, "value": float(n["value"][i])})
        else:
            out.append(
                {
                    "is_leaf": False,
                    "feature_idx": int(n["feature_idx"][i]),
                    "threshold": float(n["num_threshold"][i]),
                    "left": int(n["left"][i]),
                    "right": int(n["right"][i]),
                    "missing_go_to_left": bool(n["missing_go_to_left"][i]),
                }
            )
    return out


def _baseline(model) -> float:
    """Constant the trees are added to; zero for a fresh HGB fit."""
    b = getattr(model, "_baseline_prediction", None)
    if b is None:
        return 0.0
    arr = np.asarray(b, dtype=np.float64).ravel()
    return float(arr[0]) if arr.size else 0.0


def hgb_to_dict(
    model,
    feature_names: list[str],
    kind: str = "classifier",
    metadata: dict | None = None,
) -> dict:
    """Convert a fitted HGB estimator to a JSON-serialisable document."""
    if kind not in ("classifier", "regressor"):
        raise ValueError(f"kind must be classifier or regressor, got {kind!r}")
    raw = model._predictors
    trees = [_tree_nodes(raw[i][0]) for i in range(len(raw))]
    doc = {
        "format": JSON_FORMAT,
        "model_type": "HistGradientBoosting",
        "kind": kind,
        "n_trees": len(trees),
        "baseline": _baseline(model),
        "loss": str(model.loss),
        "learning_rate": float(model.learning_rate),
        "traversal": "if isnan(x[f]): go missing_go_to_left; else x[f] <= threshold -> left, right otherwise",
        "leaf_value": "sum of leaf values, then " + (
            "sigmoid (binary classification)" if kind == "classifier" else "identity (regression)"
        ),
        "feature_names": list(feature_names),
        "trees": trees,
    }
    if kind == "classifier":
        doc["classes"] = [0, 1]
        doc["positive_class"] = 1
    if metadata:
        doc["metadata"] = metadata
    return doc


def save_json(model, path: Path, feature_names: list[str], kind: str, metadata: dict | None = None) -> Path:
    doc = hgb_to_dict(model, feature_names, kind=kind, metadata=metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def predict_raw_json(doc: dict, X: np.ndarray) -> np.ndarray:
    """Re-implementation of sklearn's traversal, for verification and reference."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    out = np.full(n, float(doc.get("baseline", 0.0)), dtype=np.float64)
    for tree in doc["trees"]:
        root = tree[0]
        for row in range(n):
            node = root
            while not node["is_leaf"]:
                v = X[row, node["feature_idx"]]
                if v != v:  # NaN
                    node = tree[node["left"] if node["missing_go_to_left"] else node["right"]]
                else:
                    node = tree[node["left"] if v <= node["threshold"] else node["right"]]
            out[row] += node["value"]
    return out


def predict_json(doc: dict, X: np.ndarray) -> np.ndarray:
    raw = predict_raw_json(doc, X)
    if doc.get("kind") == "classifier":
        return 1.0 / (1.0 + np.exp(-raw))
    return raw


def verify_export(model, doc: dict, X: np.ndarray, tol: float = 1e-9) -> dict[str, float]:
    """Compare the JSON evaluation against sklearn's own prediction."""
    kind = doc.get("kind", "regressor")
    if kind == "classifier":
        ref = model.predict_proba(X)[:, 1]
        got = predict_json(doc, X)
    else:
        ref = model.predict(X)
        got = predict_json(doc, X)
    ref = np.asarray(ref, dtype=np.float64)
    diff = np.abs(got - ref)
    return {
        "n": int(X.shape[0]),
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "ok": bool(diff.size == 0 or diff.max() <= tol),
    }
