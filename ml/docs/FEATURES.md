# Feature contract (C++ port)

The slip gate in `artifacts/models/speed_slip/slip_gate.json` consumes exactly
20 `float32` values per sample. This document is the contract for producing
them. It is generated from the same constants the Python code uses
(`odom_ml.data.features`), and the machine-readable half is
`artifacts/reference/feature_spec.json`, so the two cannot drift apart.

Inputs are the three permitted topics at **50 Hz** (`dt = 0.02` s):

| topic | field | meaning |
|---|---|---|
| `/vehicle/driver_position_cmd` | `position` (int8) | controller notional, −15…15, 0 = neutral |
| `/vehicle/front_bogie_velocity` | `velocity` | front bogie wheel speed |
| `/vehicle/rear_bogie_velocity` | `velocity` | rear bogie wheel speed |

---

## 0. Read this first: the three traps

**1. Units.** The recorded bags are in **m/s**, so `wheel_scale = 1.0`. The jury
answered "km/h" when asked. If the hidden test bags really are km/h, the scale
becomes 3.6. This factor multiplies both wheel topics before anything else, and
being wrong about it makes the speed output wrong by 3.6×. It is the single most
dangerous number in the pipeline. Detect it once at start-up rather than
hard-coding it blind — during the first seconds GNSS is available and the ratio
`v_wheel / v_gnss` settles near 1.0 (m/s) or 3.6 (km/h).

**2. The `v_std` window is 128 samples, not 100.** `STD_WIN` is nominally 2.0 s
= 100 samples, but the implementation rounds the window up to the next power of
two, so the effective window is **128 samples = 2.56 s**. A port that uses 100
samples will not reproduce `v_std`. It is also *population* variance
(`E[x²] − E[x]²`, divisor always 128) over a window **zero-padded at the head**,
so the first samples are biased low. Reproduce that bias; it is part of the
contract.

**3. The first sample has no slope.** The acceleration features need `x[n−1]`.
Offline the reference sets `d[0] = d[1]`, so sample 0 equals the first
computable slope. A streaming port cannot look ahead: emit `0` (or hold) for
`n = 0` and expect a difference confined to that one sample.

---

## 1. Recursive filters

All filtering is a first-order IIR:

```
y[0] = x[0]
y[n] = a * y[n-1] + b * x[n]        for n >= 1
```

`y[0] = x[0]` — the first sample **passes through unchanged**, it is not ramped
from zero. The coefficients are pre-computed so no `exp()` is needed at runtime:

| filter | τ (s) | a | b |
|---|---|---|---|
| `v_fast` | 0.2 | 0.904837418 | 0.095162582 |
| `v_slow` | 2.0 | 0.990049834 | 0.009950166 |
| `v_long` | 20.0 | 0.999000500 | 0.000999500 |
| `a_wheel` | 0.4 | 0.951229425 | 0.048770575 |
| `a_fast` | 0.6 | 0.967216100 | 0.032783900 |
| `du` | 0.2 | 0.904837418 | 0.095162582 |

The derivative chain is always **backward** difference followed by an IIR — no
centred difference anywhere, because centred looks into the future:

```
d[n] = (x[n] - x[n-1]) / dt        then IIR(d, tau)
```

`v_std` is a trailing window of 128 samples, head zero-padded, divisor fixed at
128. Keep a ring buffer of 128 `v_wheel` values per stream.

---

## 2. NaN handling

The raw topics contain gaps — a missing reading arrives as `null` in the
exported reference inputs and must be read back as **NaN, not 0**.

```
v_wheel = mean(v_front, v_rear)   if both finite
        = the finite one           if only one is finite
        = 0.0                       if neither is finite
v_diff  = v_front - v_rear         if both finite, else 0.0
```

NaN is replaced by `0.0` **before** any IIR, so no NaN can enter a recursion.
Consequence: **every emitted feature is finite**, and the exported feature
matrix contains no nulls. What was missing is carried by `vf_valid`, `vr_valid`
and `gap_front` / `gap_rear` instead.

The trees also carry a `missing_go_to_left` flag on every internal node. With
finite features that branch is unreachable — implement it as a guard anyway,
because on a NaN you get undefined behaviour rather than a wrong number.

`gap_front` / `gap_rear` = samples since the last valid reading of that bogie,
`0` when the current sample is valid, `i+1` if no valid sample has been seen yet.

---

## 3. Clamping

| quantity | clamp | why |
|---|---|---|
| `a_wheel`, `a_fast` | ±2.0 m/s² | dropout recovery steps many m/s in one 20 ms sample, differentiating to hundreds of m/s² (2.4 % of a run); a tram does ≈0.7 m/s² in service |
| `du` | ±60 notional/s | same artefact on the controller command |

The features are clamped, **not** the raw inputs.

---

## 4. Output vector

20 values, `float32`, in this order:

| # | name | formula | unit |
|---|---|---|---|
| 0 | `v_wheel` | mean of the valid bogies, else the valid one, else 0 | m/s |
| 1 | `v_front` | scaled input, 0.0 where not finite | m/s |
| 2 | `v_rear` | scaled input, 0.0 where not finite | m/s |
| 3 | `v_diff` | bogie difference when both finite, else 0 | m/s |
| 4 | `v_fast` | IIR(`v_wheel`, τ=0.2) | m/s |
| 5 | `v_slow` | IIR(`v_wheel`, τ=2.0) | m/s |
| 6 | `v_long` | IIR(`v_wheel`, τ=20.0) | m/s |
| 7 | `a_wheel` | clamp(IIR(slope(`v_wheel`), τ=0.4), ±2) | m/s² |
| 8 | `a_fast` | clamp(IIR(slope(`v_fast`), τ=0.6), ±2) | m/s² |
| 9 | `v_std` | population std of `v_wheel`, trailing 128 | m/s |
| 10 | `u` | notional, NaN → 0 | notional |
| 11 | `u_abs` | `abs(u)` | notional |
| 12 | `u_tractive` | `u * moving` | notional |
| 13 | `u_brake` | `−min(u, 0) * moving` | notional |
| 14 | `du` | clamp(IIR(slope(`u`), τ=0.2), ±60) | notional/s |
| 15 | `moving` | `v_wheel > 0.5 ? 1 : 0` | 0/1 |
| 16 | `vf_valid` | finite(`v_front`) | 0/1 |
| 17 | `vr_valid` | finite(`v_rear`) | 0/1 |
| 18 | `gap_front` | samples since last valid front reading | samples |
| 19 | `gap_rear` | samples since last valid rear reading | samples |

### Order of operations (this matters)

1. scale `v_front`, `v_rear` by `wheel_scale`
2. derive `vf_valid` / `vr_valid` from the scaled values
3. `v_wheel`, `v_diff`
4. update `v_fast`, `v_slow`, `v_long` from `v_wheel`
5. update the `a_wheel` slope+IIR from `v_wheel`
6. update the `a_fast` slope+IIR from **`v_fast`**, not from `v_wheel`
7. push `v_wheel` into the 128-sample ring buffer, emit `v_std`
8. `moving` from **this** sample's `v_wheel`, then `u_tractive` / `u_brake`
9. `du` slope+IIR from `u`
10. advance `gap_front` / `gap_rear`
11. clamp `a_wheel`, `a_fast`, `du`

---

## 5. State

Filter state is per stream and initialised at the first sample. The reference
case is the **first** n samples of a run with fresh state — a port that has been
running has different state, so only compare where both sides start from a known
state.

State to keep: 6 IIR accumulators, 3 previous raw values (`v_wheel`, `v_fast`,
`u`) for the backward differences, 2 gap counters, and a 128-entry ring buffer.
That is the entire model — no lookahead, no windowing over future samples.

---

## 6. Verifying the port

`artifacts/reference/reference_case_small.json` (500 rows) and
`reference_case_full.json` (4000 rows) contain, for one real run
(`30618_01f73500`):

* `inputs` — the raw `u`, `v_front`, `v_rear` arrays, with `null` for gaps;
* `expected.features` — the 20 columns;
* `expected.slip_probability` and `expected.slip_gate_fired` — the model output.

Check in two stages, so a failure tells you which half is wrong:

1. recompute the 20 features from `inputs`, compare to `expected.features`
   — tolerance `1e-6` absolute (features are `float32` on disk; the reference
   reproduces to `2.4e-7`);
2. feed your features through the `slip_gate.json` walk and compare to
   `expected.slip_probability` — tolerance `1e-9`; the Python evaluation of the
   same JSON matches sklearn to `2.2e-16`.

The gate fires when `slip_probability >= 0.5`. Measured behaviour on unseen runs:
recall 0.999, precision 0.811, AUC ≈ 0.65. The gate exists to *distrust* a
slipping or locked wheel, not to correct it — see the note in §7.

---

## 7. What the model is and is not for

The speed **corrector is disabled**. Validation selected `shrink = 0.0`: the raw
wheel reading is already at 0.095 m/s RMSE, and every learned correction made it
worse (0.107 m/s at full strength). `speed_residual.json` is exported for
diagnostics only — do not apply it at inference.

The slip gate is the live model. Its job is to lower trust in the wheel channel
and fall back on the model, which is what criterion 3 of the case asks for. With
recall 0.999 it is a safety gate, not a precision estimator.
