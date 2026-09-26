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


## Training status: target confirmed fixed, pending clean multi-run data

The root cause of the earlier 10.6 m/s^2 moving RMSE was **not** the model and
**not** the sample count. It was the target.

eatures_*.csv (4 dumps) compute the reference acceleration from differentiated
wheel speed, then hard-clip it. Measured on cc9e7a2:

- target std 1.41 m/s^2, range -3.74..+4.14
- implied jerk p99 56-62 m/s^3, max 104 m/s^3
- 21.8% of rows outside the +/-1.50 limit
- the reference itself pinned at exactly +/-1.6 m/s^2

eatures_final_dump.csv is the fixed pipeline. Same 16 features, same contract:

| | 4 old dumps | final_dump |
|---|---|---|
| jerk p99 | 56.7 / 62.5 | **7.1 m/s^3** |
| rows outside +/-1.50 | 21.8% | **0.00%** |
| reference clamp | +/-1.6 | +/-2.0 |

### Held-out metrics

5 files, 5-fold GroupKFold grouped by run (cross-run, but 4/5 of the training
data is the old broken target):

| regime | n | MAE | RMSE | bias | baseline RMSE |
|---|---|---|---|---|---|
| all | 76 533 | 0.342 | 0.515 | -0.010 | 0.868 |
| standstill | 45 965 | 0.193 | 0.347 | +0.005 | 0.700 |
| moving | 30 568 | 0.566 | 0.695 | -0.033 | 1.067 |

eatures_final_dump.csv alone, 5-fold leave-one-time-block-out (single run, so
this is weaker evidence than a cross-run split, but the data is clean):

| regime | n | MAE | RMSE | bias | baseline RMSE |
|---|---|---|---|---|---|
| all | 56 415 | 0.151 | 0.287 | +0.006 | 0.891 |
| standstill | 31 820 | 0.095 | 0.188 | +0.013 | 0.753 |
| moving | 24 595 | 0.224 | 0.378 | -0.002 | 1.038 |

Moving RMSE 10.598 -> 0.378. Bias is within 0.002 m/s^2, so there is no
systematic error left. eatures_final_dump.csv is a **replacement**, not an
addition: mixing it with the four old dumps degrades moving RMSE from 0.378 to
0.695, because the broken targets contaminate training.

Next step is 20-30 runs regenerated in the fixed format, so the split can be
cross-run again. Nothing else is blocking.

	arget_log_scale and 	arget_mu are NaN in every dump, so those two output
rows are exported as zeros. They are not trained.
