# C++ handoff

Backup odometry, Python side. Everything the C++ node needs is in this
repository; nothing here requires the bag dataset or any Python at runtime.

## What is ready

| Artefact | Path | Size | What it is |
|---|---|---|---|
| Slip gate | `artifacts/models/speed_slip/slip_gate.json` | 112 KB | 200 gradient-boosted trees, depth 2. Probability that the wheel reading is untrustworthy |
| Route | `artifacts/route/route_utm.json` | 494 KB | The alignment as a flat UTM polyline: `e, n, s, tangent, normal, z` per point |
| Feature spec | `ml/docs/FEATURES.md` | ~9 KB | The written contract for the 20 features |
| Machine spec | `artifacts/reference/feature_spec.json` | 7 KB | Same, machine-readable |
| Reference case (fast) | `artifacts/reference/reference_case_small.json` | 60 KB | 500 rows: raw inputs, expected features, expected probability |
| Reference case (full) | `artifacts/reference/reference_case_full.json` | 780 KB | 4000 rows, same |
| Ground truth | `artifacts/reference/ground_truth_30618_efb92709.json` | 4.8 MB | One run's reference trajectory in the jury frame, for self-scoring |

The route polyline means you do **not** need UTM, MGRS, an ellipsoid, a KD-tree
or the pathgraph registration. Find the nearest segment, walk it, done.

## Where to start

1. Implement `build_features` from `ml/docs/FEATURES.md`.
2. Check it against `reference_case_small.json`: recompute the 20 columns from
   the stored `inputs` and compare with `expected.features`. The Python
   reference reproduces itself to **2.4e-7**; your tolerance is 1e-6.
3. Only then walk `slip_gate.json` and compare against
   `expected.slip_probability` (tolerance 1e-9; Python matches to 2.2e-16).
4. Once both stages pass, integrate with `route_utm.json`.

Two stages on purpose: stage 2 failing tells you the filters are wrong, stage 3
failing tells you the tree walk is wrong, and neither masks the other.

## What to defer

- **ONNX** — not needed. A depth-2 ensemble is a list of thresholds; the JSON is
  evaluated in ~30 lines with no runtime dependency. It matters for neural nets,
  not for 200 stumps.
- **`speed_residual.json`** — the speed corrector is **disabled**. Validation
  selected `shrink = 0.0`: the raw wheel reading is already 0.095 m/s RMSE and
  every learned correction made it worse (0.107 m/s at full strength). The file
  is exported for diagnostics; do not apply it.
- **Slip diagnostics channel** — the case says it is welcome, not required.

## Things that will bite you

**Units.** The recorded bags are in **m/s**. The jury answered "km/h" when asked.
`wheel_scale` multiplies both wheel topics before anything else; wrong by 3.6 and
your speed is wrong by 3.6×. Detect it at start-up while GNSS is still available
rather than hard-coding it.

**`v_std` uses a 128-sample window, not 100.** The nominal window is 2.0 s = 100
samples but it is rounded up to the next power of two, so the effective window is
**128 samples = 2.56 s**. Population variance (`E[x²] − E[x]²`), divisor always
128, window zero-padded at the head so the first samples are biased low. Reproduce
the bias.

**The first sample has no slope.** The acceleration features need `x[n−1]`. The
offline reference sets `d[0] = d[1]`; a streaming port cannot look ahead, so emit
0 (or hold) at `n = 0` and expect a difference confined to that one sample.

**NaN handling is not optional.** The raw topics contain gaps, written as `null`
in the reference inputs and read back as NaN — do not coerce to 0. `v_wheel`
falls back to whichever bogie is valid, then to 0. Features come out finite, but
every internal tree node carries a `missing_go_to_left` flag; implement the branch
as a guard, because on a NaN you get undefined behaviour rather than a wrong
number.

**Do not forget `baseline`.** Summing the leaves gives a logit; `baseline` is the
starting offset (1.3148 for the gate) and the sigmoid comes after.

**The registration is not settled.** The route transform differs by 0.164° between
fits on different subsets of runs. Do not treat `ground_truth_*.json` or the route
as absolute truth — they are self-consistent, not certified. Relative behaviour is
meaningful; absolute offsets may shift once the official map arrives (the jury
said it will be sent separately).

## Checking yourself

Recompute the features from the stored inputs and compare. The files also carry
`implementation_notes` with exact IIR coefficients, the update order, the units
table and the NaN rules, so you should not need to read any Python.

## Note on the case requirements

The jury scores real-time behaviour, so publish online only: `/result/velocity`
(`VelocitySensor.velocity`, m/s) and `/result/position` (`Odometry.pose.pose.position`),
≥10 Hz, input-to-output latency ≤100 ms, with `header.stamp` taken from the bag
clock and not from wall time. Batch post-processing is not accepted. GNSS is
allowed for initial alignment and for scoring, never in the estimation loop.
