# ML contract

The interface between the Python training side and the C++ runtime. Everything
here is enforced by `ml/scripts/train_speed_residual.py` and the tests; nothing
is recomputed on both sides.

Training input is the C++ feature dump and nothing else. Deriving the features a
second time in Python is how the two implementations would drift apart, so the
dump is the contract.

## Artefacts

| File | Contents |
|---|---|
| `models/ml_model.yaml` | Descriptor: order, normalisation, limits, provenance, metrics |
| `models/speed_residual.bin` | 51 little-endian float64, `[W (3x16) row-major][b (3)]` |
| `models/validation_report.json` | Metrics and target distributions, for the write-up |
| `artifacts/route/route_utm.json` | Route as a flat UTM polyline |
| `artifacts/route_registration.json` | Pathgraph → UTM rigid transform |

`speed_residual.bin` is 408 bytes, no header. Read it as:

```
W = values[0:48].reshape(3, 16)      # row-major: row 0 -> a_residual
b = values[48:51]
```

## Applying the model

```
z = (x - norm.mean) / norm.std        # elementwise, 16 values
y = W @ z + b                         # 3 values
`

There is no clipping of z. A winsorisation at 8 sigma was tried and measured:
it moved the held-out moving RMSE by 0.008 m/s^2, so it was removed rather than
carried as a contract obligation.``

Normalisation lives in the descriptor and **must** be applied by the caller
before the matrix product. The exported weights act on standardised inputs, not
raw ones.

Then clamp the outputs to the limits from the descriptor:

| Output | Unit | Clamp |
|---|---|---|
| `a_residual` | m/s² | ±0.80 |
| `log_scale` | log | ±0.05 |
| `mu` | – | [0.05, 0.90] |

## Feature vector

Order is fixed and 0-based. `b_scale` is index 7, `slip_index` is index 8 — prose
that calls `b_scale` "column 8" is counting from 1.

| # | Name | Definition | Unit | Range |
|---|---|---|---|---|
| 0 | `u` | controller notional / 15 | – | [-1, 1] |
| 1 | `v` | filtered body speed | km/h | ≥ 0 |
| 2 | `a_model` | physics model, no ML residual | m/s² | ~±1.2 |
| 3 | `grade` | path grade from the map, 0.0 while the map is off | rad | [-0.06, 0.06] |
| 4 | `mu` | current adhesion estimate | – | [0.05, 0.90] |
| 5 | `omega_front` | `v_front / 0.30` | rad/s | ≥ 0 |
| 6 | `omega_rear` | `v_rear / 0.30` | rad/s | ≥ 0 |
| 7 | `b_scale` | odometry scale | – | [0.80, 1.25] |
| 8 | `slip_index` | C++ slip detector output | – | [0, 1] |
| 9 | `trust` | trust in the odometry | – | [0, 1] |
| 10 | `cmd_rate` | `d(u)/dt` | 1/s | |
| 11 | `dt` | sample period | s | ~0.02 |
| 12 | `abs_u` | `abs(u)` | – | [0, 1] |
| 13 | `u_sq` | `u²` | – | [0, 1] |
| 14 | `v_sq` | `v²` | (km/h)² | |
| 15 | `force_ratio` | `abs(F_drive - F_brake) / adhesion_limit` | – | [0, 1] |

## How it was trained

- **Backend** `Ridge` (L2), `alpha = 1.0`, `fit_intercept = true`. Dense linear
  solve, so the result is deterministic; `--seed` is recorded regardless.
- **Split by run, not by sample.** Whole runs are held out, so no window of a
  drive appears in both training and validation. A run-identifier column is
  required — the script refuses to train without one rather than silently
  splitting samples.
- **Normalisation** mean and std computed on training rows only.
- **Target clipping** applied to the targets, matching what the runtime enforces,
  so the model never sees a value it will not be asked to predict. The unclipped
  p1/p99 is reported in the descriptor under `target_report`; that is the
  evidence for whether the ±0.80 limit is right.
- **Dropping non-finite rows** is reported, never silent.

### `b_scale` leakage

`b_scale` is derived from the same odometry that `log_scale` corrects, so a
model handed it can partly read the answer back instead of predicting it. The
descriptor records `b_scale_log_scale_pearson_r`. As long as that correlation is
meaningful, treat `log_scale` as a diagnostic rather than a prediction, or ask
the C++ side to write a constant `b_scale` into the training dump.

`--drop-b-scale` removes the feature and trains on 15. It breaks the 16-feature
contract, so it is off by default.

## Route artefacts for C++

`artifacts/route/route_utm.json`, format `odom_ml.route_utm/2`, 4710 points,
~832 KB. Columns, in order:

```
x, y, z, s, heading_deg, heading_rad, grade, grade_rad, curvature, t_x, t_y, n_x, n_y
```

- `x`, `y` — UTM zone 37N easting and northing, **false northing disabled**, so
  northings are ~6 188 000 and not ~16 188 000.
- `s` — arclength from the start of the polyline, 0 to 4708.34 m.
- `heading_deg` / `heading_rad` — direction of increasing `s`, measured from
  east counter-clockwise, **unwrapped continuously**. It does not wrap at ±180°.
- `grade` — `dz/ds`, dimensionless. `grade_rad` is `atan(grade)`. Positive is
  climbing.
- `curvature` — `d(heading)/ds` in 1/m, exactly 0 on a straight section, where a
  turn radius would be infinite. Minimum turn radius on this route is 30.7 m.
- `t_x`, `t_y` — unit tangent; `n_x`, `n_y` — left normal, so a positive
  cross-track offset is left of travel.

`artifacts/route_registration.json` holds the 2x2 rotation and translation that
puts the supplied pathgraph into UTM, with the fitted residuals and scale.

**The route does not self-overlap**: the two legs are on separate tracks, so a
nearest-point arclength is unique and no search window is needed. Position from
arclength is therefore a scan over the polyline, not a search.

## Two conventions that are easy to get wrong

**The output frame is MGRS referenced to a fixed 100 km square, continuous.**
The jury's own example has `x = 103501.63`, which is greater than 100 000, so it
is *not* `easting mod 100000`. The reference square is 37UCB — column C is
300 km, so `x = easting - 300000`, which reproduces their example to 38.6 m. The
northing does not reconcile to better than 2.5 km, which is consistent with the
published coordinates and the MGRS string in their message describing different
points; treat the square as fixed and the origin as the start of each run.

**`z` is absolute ellipsoidal height**, as in `NavSatFix.altitude`, not a
difference from the start — their example 167.41 matches the absolute heights in
the dataset. `x` and `y` are relative. The mix is unusual and worth confirming
with the jury, but both readings are implemented and documented.


## Training status: TEMPORARY, not fit for the moving regime

This artefact was trained on **4 feature dumps** so the C++ integration path could
be exercised end to end. It is wired up correctly; it is not a finished model.

Held-out metrics, pooled over 4-fold GroupKFold grouped by run:

| regime | n | MAE | RMSE | bias | baseline RMSE |
|---|---|---|---|---|---|
| all | 20 118 | 2.010 | 5.781 | +1.736 | 0.799 |
| standstill (v <= 1.8 km/h) | 14 145 | 0.183 | 0.330 | -0.019 | 0.562 |
| **moving (v > 1.8 km/h)** | 5 973 | 6.337 | **10.598** | **+5.891** | 1.179 |

Standalone limit is 0.33 m/s^2, better than the constant baseline. Moving is 9x
**worse** than baseline. Do not ship this for driving.

Two independent causes, both measured:

1. **Too few moving runs.** Only 2 of 4 dumps contain motion (7bfbb5ed 481 moving
   rows, bcc9e7a2 638). Leave-one-run-out therefore trains the moving head on a
   single run, while 14 145 standstill rows pull towards a constant.

2. **The target is noise-dominated.** In run bcc9e7a2, _model is pinned at
   exactly +2.2000 m/s^2 across consecutive samples (the physics model is
   saturating), and 	arget_a_residual moves from -3.744 to -1.508 m/s^2 within
   60 ms. The implied true acceleration swings -1.54 -> +0.69 m/s^2 in 60 ms,
   which is not physically achievable for a tram. In the moving regime the target
   has std 1.41 m/s^2 and spans -3.74..+4.14; 21.8% of all rows fall outside the
   +/-1.50 clip. A corrector cannot predict a target whose noise exceeds its
   signal, so cause 1 alone will not make the moving head work.

Retrain is required once both are addressed: 20-30 moving runs, and a target
derived from a smooth, clamp-free acceleration reference.

	arget_log_scale and 	arget_mu are NaN in every dump, so those two output
rows are exported as zeros. They are not trained.
