# Learned residual corrector artifacts

This directory holds the model artifacts consumed by `MlCorrector`. It is
installed to `share/tram_odometry/models` and is the default value of
`ml.model_dir` in the node.

## What the runtime expects

* `ml_model.yaml` — descriptor. The *contract* (feature names, order, count,
  output names, schema version) is compiled into
  `include/tram_odometry/ml_features.hpp` and validated at load time. A
  mismatch is rejected and the node falls back to pure physics.
* `*.bin` — flat little-endian `float64` weights, one file, no header.
  * `linear`: `[W (3 x 16, row-major)][b (3)]` = 51 doubles.
  * `mlp`: for every layer `[W (n_out x n_in, row-major)][b (n_out)]`,
    layer sizes taken from `layer_sizes` in the descriptor.

Both layouts are produced in one line of numpy, see `docs/04_ml_contract.md`.

## Files

| File | Purpose |
|---|---|
| `ml_model.yaml` | default descriptor, `linear` backend |
| `ml_stub_linear.bin` | all-zero weights: a no-op corrector, pipeline runs |

## Rules for anything committed here

1. Only trained on the allowed train bags. Never on the verification bags.
2. `tools/validate_artifact.py` must pass before a model is committed.
3. The descriptor carries the training metrics and the data revision, so the
   jury can see which model produced which numbers.
4. Weights are small (a few kB). If a model ever needs real ONNX Runtime, the
   `.onnx` file goes here too with `backend: onnx`, and the package has to be
   built with `-DTRAM_WITH_ONNXRUNTIME=ON`.
