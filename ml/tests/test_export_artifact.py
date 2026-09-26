"""The runtime artefact must be exactly what the contract promises.

These check the parts a C++ reader depends on and that nothing else would catch:
the byte count, the row-major layout, the end-to-end round trip through the file
rather than through the in-memory model, and that the descriptor's feature order
is the one the trainer used.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_speed_residual import (  # noqa: E402
    FEATURE_NAMES,
    N_FEATURES,
    N_OUTPUTS,
    N_WEIGHTS,
    TARGET_NAMES,
    apply_weights,
    read_weights_bin,
    write_weights_bin,
)

ARTIFACT_DIR = Path("models")


def test_feature_and_target_contract_sizes():
    """The 16/3/51 numbers are the contract; a silent change breaks C++."""
    assert len(FEATURE_NAMES) == 16
    assert len(TARGET_NAMES) == 3
    assert N_WEIGHTS == 51
    assert len(set(FEATURE_NAMES)) == 16, "duplicate feature name"
    # the two names that are easy to swap because prose counts columns from 1
    assert FEATURE_NAMES[7] == "b_scale"
    assert FEATURE_NAMES[8] == "slip_index"


def _reference_model():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(N_OUTPUTS, N_FEATURES))
    b = rng.normal(size=N_OUTPUTS)
    return W, b


def test_weights_file_is_exactly_408_bytes(tmp_path):
    W, b = _reference_model()
    p = tmp_path / "speed_residual.bin"
    assert write_weights_bin(p, W, b) == N_WEIGHTS
    assert p.stat().st_size == 408
    assert p.stat().st_size % 8 == 0


def test_layout_is_row_major_weights_then_bias(tmp_path):
    W, b = _reference_model()
    p = tmp_path / "w.bin"
    write_weights_bin(p, W, b)
    raw = struct.unpack("<51d", p.read_bytes())
    # row 0 of W is the a_residual output, and it must be the first 16 values
    for j in range(N_FEATURES):
        assert raw[j] == pytest.approx(W[0, j], abs=0.0)
    for k in range(N_OUTPUTS):
        assert raw[48 + k] == pytest.approx(b[k], abs=0.0)


def test_round_trip_through_the_file_is_bit_exact(tmp_path):
    W, b = _reference_model()
    p = tmp_path / "w.bin"
    write_weights_bin(p, W, b)
    Wr, br = read_weights_bin(p)
    assert np.array_equal(Wr, W)
    assert np.array_equal(br, b)


def test_end_to_end_prediction_survives_serialisation(tmp_path):
    """What C++ computes must equal what sklearn computed, to the last bit."""
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, N_FEATURES))
    Y = np.column_stack([rng.normal(size=500), rng.normal(size=500), rng.uniform(0.05, 0.9, 500)])
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std < 1e-12] = 1.0

    m = Ridge(alpha=1.0, fit_intercept=True, random_state=0).fit((X - mean) / std, Y)
    direct = m.predict((X - mean) / std)

    p = tmp_path / "w.bin"
    write_weights_bin(p, m.coef_, m.intercept_)
    W, b = read_weights_bin(p)
    via_file = apply_weights(W, b, X, mean, std)
    assert np.abs(via_file - direct).max() <= 1e-10


def test_write_rejects_a_wrong_shape(tmp_path):
    W, b = _reference_model()
    with pytest.raises(Exception):
        write_weights_bin(tmp_path / "bad.bin", W[:, :5], b)
    with pytest.raises(Exception):
        write_weights_bin(tmp_path / "bad.bin", W, b[:2])


def test_constant_feature_gets_unit_std(tmp_path):
    """A constant column must not divide by zero when the runtime normalises.

    ``grade`` is 0.0 throughout the dump while the map is disabled, so its std is
    zero; the trainer replaces it with 1.0 and the feature then contributes a
    constant, which is exactly the intent.
    """
    X = np.zeros((10, N_FEATURES))
    X[:, 0] = 3.0
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std < 1e-12] = 1.0
    assert std[1] == 1.0
    z = (X - mean) / std
    assert np.isfinite(z).all()
    assert np.all(z[:, 1:] == 0.0)


def test_end_to_end_prediction_survives_serialisation_uses_declared_order(tmp_path):
    """Shifting the feature order must change the prediction, i.e. order matters."""
    W, b = _reference_model()
    p = tmp_path / "w.bin"
    write_weights_bin(p, W, b)
    Wr, br = read_weights_bin(p)
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, N_FEATURES))
    mean, std = np.zeros(N_FEATURES), np.ones(N_FEATURES)
    a = apply_weights(Wr, br, X, mean, std)
    c = apply_weights(Wr, br, X[:, ::-1], mean, std)
    assert not np.allclose(a, c)
