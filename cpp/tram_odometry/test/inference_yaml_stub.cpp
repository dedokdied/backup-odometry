// Link-time stubs for the descriptor loader, so the blind-drift probe can be
// built without yaml-cpp. The learned corrector is disabled by default
// (Params::corrector.enable == false), so it never runs in this probe and the
// estimator path under measurement is the pure physics/wheel path. Returning
// valid == false makes the corrector degrade to physics, which is the same
// state the probe measures.

#include <memory>
#include <string>

#include "tram_odometry/inference.hpp"
#include "tram_odometry/ml_features.hpp"

namespace tram {

ModelDescriptor load_descriptor(const std::string&) {
  ModelDescriptor d;
  d.valid = false;
  d.error = "stub: yaml-cpp not linked into the drift probe";
  return d;
}

ModelDescriptor parse_descriptor(const std::string&) {
  ModelDescriptor d;
  d.valid = false;
  d.error = "stub: yaml-cpp not linked into the drift probe";
  return d;
}

FeatureNorm ModelDescriptor::norm() const { return FeatureNorm::identity(); }

std::unique_ptr<IInferenceBackend> make_backend(const ModelDescriptor&,
                                                const std::string&) {
  return nullptr;
}

bool onnx_backend_available() { return false; }

}  // namespace tram
