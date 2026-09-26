// Residual corrector: physics model + learned correction, with a watchdog.
//
// The physics model (LongitudinalModel) is always in charge of the state
// propagation. This class only produces a *bounded* correction to the model
// acceleration and to the odometry scale, and it is allowed to do nothing at
// all. Guarantees, in order of importance:
//
//   1. It never breaks the estimate. Non-finite inputs or outputs, an over-budget
//      call, a corrupt artifact or a backend that throws are all turned into
//      "no correction this cycle" plus a diagnostic counter.
//   2. It is bounded. a_residual and log_scale are clipped to the limits from
//      the descriptor, so a pathological model degrades the estimate slightly
//      instead of diverging it.
//   3. It is cheap and allocation free in the control loop.
#pragma once

#include <memory>
#include <string>

#include "tram_odometry/inference.hpp"
#include "tram_odometry/ml_features.hpp"
#include "tram_odometry/params.hpp"

namespace tram {

/// Everything the corrector is allowed to see. Deliberately a flat POD: the
/// feature vector is a pure function of this struct, which keeps the contract
/// with the Python side trivially reproducible.
struct CorrectorInput {
  double t = 0.0;
  double u = 0.0;             ///< controller position, [-1, 1]
  double v = 0.0;             ///< m/s
  double a_model = 0.0;       ///< m/s^2, from the physics model
  double grade = 0.0;         ///< rad
  double mu = 0.35;           ///< -
  double omega_front = 0.0;   ///< rad/s
  double omega_rear = 0.0;    ///< rad/s
  double b_scale = 1.0;       ///< -
  double slip_index = 0.0;    ///< [0, 1]
  double trust = 0.0;         ///< [0, 1]
  double cmd_rate = 0.0;      ///< 1/s
  double dt = 0.02;           ///< s
  double drive_force = 0.0;   ///< N
  double brake_force = 0.0;   ///< N
  double adhesion_limit = 1.0;  ///< N
};

struct CorrectorOutput {
  double a_residual = 0.0;  ///< m/s^2, add to the model acceleration
  double log_scale = 0.0;   ///< multiply b_scale by exp(log_scale)
  double mu_model = 0.0;    ///< adhesion estimate from the model
  bool applied = false;     ///< false => physics only
  bool degraded = false;    ///< backend disabled or failing
  double inference_ms = 0.0;
};

class MlCorrector {
 public:
  MlCorrector();
  ~MlCorrector();

  MlCorrector(const MlCorrector&) = delete;
  MlCorrector& operator=(const MlCorrector&) = delete;

  /// Loads the descriptor and the weights. Safe to call with ml.enable == false
  /// or with a missing directory: the corrector then simply reports "no model".
  void configure(const MlParams& params, const std::string& model_dir);

  /// One correction step. Never throws, never allocates.
  CorrectorOutput step(const CorrectorInput& in);

  /// Fills a feature vector exactly as step() does (exposed for the Python
  /// side to reproduce the same numbers when generating training data).
  static void buildFeatures(const CorrectorInput& in, FeatureVector* out);

  bool hasModel() const { return has_model_; }
  bool degraded() const { return degraded_; }
  InferenceStats stats() const { return stats_; }
  const ModelDescriptor& descriptor() const { return desc_; }
  const std::string& statusLine() const { return status_; }

  void reset();

 private:
  MlParams p_;
  ModelDescriptor desc_;
  FeatureNorm norm_;
  std::unique_ptr<IInferenceBackend> backend_;
  InferenceStats stats_;
  CorrectorOutput last_;
  bool has_model_ = false;
  bool degraded_ = false;
  int consecutive_failures_ = 0;
  double last_u_ = 0.0;
  double last_t_ = -1e9;
  std::string status_ = "disabled";

  // scratch, sized once in configure()
  FeatureVector raw_;
  FeatureVector scaled_;
  double out_[kNumOutputs] = {};

  void disable(const char* reason);
};

}  // namespace tram
