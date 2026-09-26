// Learned-model inference backends for the residual corrector.
//
// Design constraints that come from the task statement:
//   * the package must build with `colcon build` and *no external internet*, so
//     there is no build-time dependency on ONNX Runtime or any ML framework;
//   * the node must be robust: a missing, corrupt, NaN-producing or simply
//     absent model may never take the odometry down.
//
// Therefore the runtime supports two families of backends:
//   * `linear` / `mlp` — a tiny dense evaluator implemented right here in C++17
//     on top of a flat little-endian float64 weight file. Zero dependencies,
//     single-digit microseconds per call, and trivially reproducible in numpy,
//     so the Python side can emit a working artifact in one line.
//   * `onnx` — optional, compiled only when ONNX Runtime is found at configure
//     time (TRAM_WITH_ONNXRUNTIME). Anything more exotic (GDBT, RNN, ensembles)
//     goes through this path.
//
// Every backend is wrapped by a watchdog in MlCorrector; on any failure the
// analytic physics model remains the source of truth.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tram_odometry/ml_features.hpp"

namespace tram {

/// Runtime health of the inference path, published in /result/diagnostics.
struct InferenceStats {
  uint64_t calls = 0;
  uint64_t failures = 0;
  uint64_t rejected = 0;   ///< non-finite model output, clipped away
  uint64_t overruns = 0;   ///< calls slower than the per-call budget
  double mean_ms = 0.0;
  double max_ms = 0.0;
  bool degraded = false;   ///< backend disabled for the rest of the run
  std::string backend;     ///< "linear" | "mlp" | "onnx" | "null"
  std::string model_id;
};

/// Abstract dense regressor: features in, kNumOutputs corrections out.
class IInferenceBackend {
 public:
  virtual ~IInferenceBackend() = default;

  /// Short identifier for diagnostics.
  virtual const char* name() const = 0;

  /// One forward pass. Returns false if the backend is unusable; the caller
  /// then falls back to pure physics. Must not allocate.
  virtual bool run(const double* features, int n_features, double* outputs,
                   int n_outputs) = 0;

  virtual bool ready() const = 0;
  virtual const std::string& lastError() const = 0;
};

/// Affine model  y = W x + b.  The workhorse: a ridge residual on top of
/// physics, and the shape a gradient-boosted model is distilled into.
class LinearBackend final : public IInferenceBackend {
 public:
  LinearBackend(int n_features, int n_outputs);
  const char* name() const override { return "linear"; }

  /// row-major W: n_outputs x n_features
  void setWeights(std::vector<double> w, std::vector<double> b);

  bool run(const double* features, int n_features, double* outputs,
           int n_outputs) override;
  bool ready() const override { return ready_; }
  const std::string& lastError() const override { return error_; }

 private:
  int n_in_ = 0;
  int n_out_ = 0;
  std::vector<double> w_, b_;
  std::vector<double> y_;
  bool ready_ = false;
  std::string error_;
};

/// Fully connected net, tanh hidden activations, linear output.
/// Layer sizes come from the descriptor; weights are the concatenation of
/// (W, b) per layer, row-major.
class MlpBackend final : public IInferenceBackend {
 public:
  MlpBackend(std::vector<int> layer_sizes);
  const char* name() const override { return "mlp"; }

  /// Flat weights: for every layer, n_in*n_out doubles then n_out biases.
  void setWeights(std::vector<double> flat, size_t expected);

  bool run(const double* features, int n_features, double* outputs,
           int n_outputs) override;
  bool ready() const override { return ready_; }
  const std::string& lastError() const override { return error_; }
  size_t layers() const { return layer_sizes_.size(); }

 private:
  std::vector<int> layer_sizes_;
  std::vector<double> flat_;
  std::vector<double> buf_a_, buf_b_;
  bool ready_ = false;
  std::string error_;
};

/// Always returns zeros: the corrector is disabled, physics rules.
class NullBackend final : public IInferenceBackend {
 public:
  const char* name() const override { return "null"; }
  bool run(const double*, int, double* outputs, int n_outputs) override {
    for (int i = 0; i < n_outputs; ++i) outputs[i] = 0.0;
    return true;
  }
  bool ready() const override { return true; }
  const std::string& lastError() const override { return error_; }

 private:
  std::string error_;
};

/// Everything the runtime needs to know about a model artifact.
struct ModelDescriptor {
  bool valid = false;
  std::string error;

  int schema_version = 0;
  std::string model_id = "unknown";
  std::string created_utc;
  std::string backend = "null";  ///< linear | mlp | onnx
  std::string weights_file;      ///< relative to the descriptor directory
  std::vector<std::string> features;
  std::vector<std::string> outputs;
  std::vector<int> layer_sizes;  ///< mlp only, includes input and output size
  std::vector<double> norm_mean;
  std::vector<double> norm_std;
  OutputLimits limits;
  /// Optional ONNX graph name, used by the optional onnx backend.
  std::string onnx_model_file;
  std::string onnx_input_name;
  std::string onnx_output_name;

  FeatureNorm norm() const;
};

/// Parses a YAML descriptor and validates it against the compiled-in schema
/// (feature names/order/count, output names/order/count, schema version, finite
/// normalisation, sane layer sizes). Never throws, never aborts: on any problem
/// it returns valid == false with a human-readable reason.
ModelDescriptor load_descriptor(const std::string& path);

/// Same validation for an already parsed YAML node (used by tests).
ModelDescriptor parse_descriptor(const std::string& yaml_text);

/// Instantiates the backend described by `desc`, loading the weight file from
/// `dir`. Returns a NullBackend (never nullptr) when the artifact is unusable.
std::unique_ptr<IInferenceBackend> make_backend(const ModelDescriptor& desc,
                                                const std::string& dir);

/// True when the package was built with the optional ONNX Runtime backend.
bool onnx_backend_available();

/// Reads a flat little-endian float64 file. Returns false on any size mismatch.
bool read_f64_file(const std::string& path, size_t expected, std::vector<double>* out);

}  // namespace tram
