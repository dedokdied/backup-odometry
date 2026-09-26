// Feature and output schema of the learned residual corrector.
//
// THIS HEADER IS THE CONTRACT. tools/validate_artifact.py parses the
// kFeatureNames / kOutputNames arrays below and rejects any model artifact
// whose feature order, feature count, output order or schema version differs.
// Nothing in the pipeline may reorder or rename these without bumping
// kFeatureSchemaVersion and updating docs/04_ml_contract.md.
//
// The corrector is deliberately a *residual* on top of the physics model, not a
// replacement for it: the analytic model owns the state propagation and the ML
// part only learns what physics gets wrong (unmodelled losses, driver habit,
// weather, wheel wear, mass error). That keeps the solution stable when the
// model is fed inputs far outside its training distribution.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace tram {

/// Bump on any incompatible change of the feature vector or output vector.
inline constexpr int kFeatureSchemaVersion = 1;

inline constexpr int kNumFeatures = 16;
inline constexpr int kNumOutputs = 3;

/// Indices into the feature vector. Order is part of the contract.
enum FeatureIndex : int {
  kFControllerPos = 0,  ///< u, normalised driver handle position, [-1, 1]
  kFSpeed = 1,          ///< v, filtered body speed, m/s
  kFAccelModel = 2,     ///< a predicted by the physics model, m/s^2
  kFGrade = 3,          ///< path inclination from the map, rad
  kFMu = 4,             ///< current adhesion estimate, -
  kFOmegaFront = 5,     ///< front bogie angular rate, rad/s
  kFOmegaRear = 6,      ///< rear bogie angular rate, rad/s
  kFOdomScale = 7,      ///< b_scale from the observer, -
  kFSlipIndex = 8,      ///< slip detector output, [0, 1]
  kFTrust = 9,          ///< odometry trust from the slip detector, [0, 1]
  kFCmdRate = 10,       ///< d(u)/dt, 1/s
  kFDt = 11,            ///< control period, s
  kFAbsU = 12,          ///< |u|, separates the tractive and brake branches
  kFUSq = 13,           ///< u^2, cheap curvature of the handle characteristic
  kFVSq = 14,           ///< v^2, where the aerodynamic loss becomes quadratic
  kFForceRatio = 15,    ///< |F_drive - F_brake| / adhesion limit, [0, 1]
};

/// Canonical feature names, index-aligned with FeatureIndex.
inline constexpr const char* kFeatureNames[kNumFeatures] = {
    "u",           "v",          "a_model",         "grade",
    "mu",          "omega_front", "omega_rear",     "b_scale",
    "slip_index",  "trust",      "cmd_rate",        "dt",
    "abs_u",       "u_sq",       "v_sq",            "force_ratio",
};

/// Indices into the output vector. Order is part of the contract.
enum OutputIndex : int {
  kOAresidual = 0,  ///< additive correction of the model acceleration, m/s^2
  kOLogScale = 1,   ///< multiplicative correction of the odometry scale, log units
  kOMu = 2,         ///< model-based adhesion estimate, -
};

/// Canonical output names, index-aligned with OutputIndex.
inline constexpr const char* kOutputNames[kNumOutputs] = {
    "a_residual",
    "log_scale",
    "mu",
};

/// Physical plausibility envelope for a residual. A model that leaves this
/// envelope is a broken model, whatever its training RMSE says.
///
/// Raised from 0.80 to 1.50 m/s^2 after measuring the training labels: on the
/// real bags roughly a quarter of them exceed 0.80, so the old limit threw away
/// a large part of the signal and the runtime clipped about half of what the
/// model produced. Whatever this value is, the training target must be clipped
/// to the SAME number, otherwise the model learns a distribution it never gets
/// to apply.
inline constexpr double kDefaultMaxAccelResidual = 1.50;  ///< m/s^2
inline constexpr double kDefaultMaxLogScale = 0.05;        ///< ~5 % on the scale
inline constexpr double kDefaultMuMin = 0.05;
inline constexpr double kDefaultMuMax = 0.90;

/// Fixed-capacity raw feature buffer: no heap traffic in the control loop.
struct FeatureVector {
  double x[kNumFeatures] = {};

  double& operator[](int i) { return x[i]; }
  double operator[](int i) const { return x[i]; }
  void setZero() {
    for (int i = 0; i < kNumFeatures; ++i) x[i] = 0.0;
  }
  bool finite() const {
    for (int i = 0; i < kNumFeatures; ++i) {
      if (!std::isfinite(x[i])) return false;
    }
    return true;
  }
};

/// Standardisation applied before the network. mean/std come from the training
/// set and travel inside the model descriptor, never hard-coded here.
struct FeatureNorm {
  double mean[kNumFeatures] = {};
  double inv_std[kNumFeatures] = {};

  /// identity = no standardisation (a linear model may not need it)
  static FeatureNorm identity() {
    FeatureNorm n;
    for (int i = 0; i < kNumFeatures; ++i) {
      n.mean[i] = 0.0;
      n.inv_std[i] = 1.0;
    }
    return n;
  }

  void apply(const FeatureVector& in, FeatureVector& out) const {
    for (int i = 0; i < kNumFeatures; ++i) {
      const double s = in[i] * inv_std[i] - mean[i] * inv_std[i];
      out[i] = std::isfinite(s) ? s : 0.0;
    }
  }
};

/// Output post-processing limits.
struct OutputLimits {
  double max_a_residual = kDefaultMaxAccelResidual;
  double max_log_scale = kDefaultMaxLogScale;
  double mu_min = kDefaultMuMin;
  double mu_max = kDefaultMuMax;
};

inline double clampOutput(double v, double lo, double hi) {
  if (!std::isfinite(v)) return 0.0;
  return v < lo ? lo : (v > hi ? hi : v);
}

}  // namespace tram
