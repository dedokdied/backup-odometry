// Minimal replacement for inference.cpp, which pulls in yaml-cpp that this
// toolchain does not have. It parses only the shape that
// ml/scripts/train_speed_residual.py writes and refuses anything else rather
// than guessing.
//
// This exists so the C++ inference path can actually be exercised: the previous
// stub returned valid == false, which makes the corrector degrade to physics, so
// any measurement of the C++ path through it was measuring nothing.

#include "tram_odometry/inference.hpp"
#include "tram_odometry/ml_features.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace tram {
namespace {

std::string trim(const std::string& s) {
  const size_t a = s.find_first_not_of(" \t\r");
  if (a == std::string::npos) return "";
  const size_t b = s.find_last_not_of(" \t\r");
  return s.substr(a, b - a + 1);
}

std::string unquote(const std::string& s) {
  if (s.size() >= 2 && s.front() == '"' && s.back() == '"') return s.substr(1, s.size() - 2);
  if (s.size() >= 2 && s.front() == '\'' && s.back() == '\'') return s.substr(1, s.size() - 2);
  return s;
}

std::vector<std::string> inline_list(const std::string& s) {
  std::vector<std::string> out;
  std::string body = trim(s);
  if (!body.empty() && body.front() == '[') body = body.substr(1);
  if (!body.empty() && body.back() == ']') body.pop_back();
  std::stringstream ss(body);
  std::string item;
  while (std::getline(ss, item, ',')) {
    const std::string t = unquote(trim(item));
    if (!t.empty()) out.push_back(t);
  }
  return out;
}

// A list under `key:`, either inline [a, b] or as following "- a" lines.
bool block_list(const std::vector<std::string>& lines, size_t at, const std::string& key,
                std::vector<std::string>& out) {
  const std::string head = key + ":";
  if (at >= lines.size() || trim(lines[at]) != head) return false;
  const std::string tail = trim(lines[at].substr(head.size()));
  if (!tail.empty()) {
    out = inline_list(tail);
    return true;
  }
  for (size_t i = at + 1; i < lines.size(); ++i) {
    const std::string t = trim(lines[i]);
    if (t.rfind("- ", 0) != 0) break;
    out.push_back(unquote(trim(t.substr(2))));
  }
  return true;
}

// A numeric list nested under `key:` -> `sub:`, one value per line.
bool nested_list(const std::vector<std::string>& lines, size_t at, const std::string& key,
                 const std::string& sub, std::vector<double>& out) {
  const std::string head = key + ":";
  if (at >= lines.size() || trim(lines[at]) != head) return false;
  const std::string subhead = sub + ":";
  for (size_t i = at + 1; i < lines.size(); ++i) {
    if (trim(lines[i]) != subhead) continue;
    for (size_t j = i + 1; j < lines.size(); ++j) {
      const std::string t = trim(lines[j]);
      if (t.empty()) continue;
      if (t.find(':') != std::string::npos) break;
      // Strip the list marker first. A negative value is written as
      // "- -0.304...", and reading the first '-' as a sign fails on the second.
      std::string body = t;
      if (body.rfind("- ", 0) == 0) body = trim(body.substr(2));
      double value = 0.0;
      std::istringstream ss(body);
      if (ss >> value) out.push_back(value);
    }
    return true;
  }
  return false;
}

}  // namespace

ModelDescriptor parse_descriptor(const std::string& text) {
  ModelDescriptor d;
  std::vector<std::string> lines;
  std::stringstream ss(text);
  std::string line;
  while (std::getline(ss, line)) lines.push_back(line);

  for (size_t i = 0; i < lines.size(); ++i) {
    const std::string t = trim(lines[i]);
    // Only top-level keys. provenance repeats backend: with the value "Ridge",
    // and letting it overwrite the descriptor's own "linear" made make_backend
    // bail out with weights present and readable.
    const bool top = !lines[i].empty() && (lines[i][0] != ' ' && lines[i][0] != '\t');
    if (top && t.rfind("schema_version:", 0) == 0) {
      d.schema_version = std::atoi(t.c_str() + 16);
    } else if (top && t.rfind("model_id:", 0) == 0) {
      d.model_id = unquote(trim(t.substr(9)));
    } else if (top && t.rfind("backend:", 0) == 0) {
      d.backend = unquote(trim(t.substr(8)));
    } else if (top && t.rfind("weights_file:", 0) == 0) {
      d.weights_file = unquote(trim(t.substr(13)));
    } else {
      std::vector<std::string> names;
      std::vector<double> nums;
      if (block_list(lines, i, "features", names)) {
        d.features = names;
      } else if (block_list(lines, i, "outputs", names)) {
        d.outputs = names;
      } else if (trim(lines[i]) == "norm:") {
        // Both sub-lists live under one `norm:` line, so this cannot be an
        // else-if chain: the first match would short-circuit the second and
        // norm_std would stay empty.
        std::vector<double> mean_values;
        std::vector<double> std_values;
        nested_list(lines, i, "norm", "mean", mean_values);
        nested_list(lines, i, "norm", "std", std_values);
        d.norm_mean = mean_values;
        d.norm_std = std_values;
      }
    }
  }

  if (d.schema_version != kFeatureSchemaVersion) {
    d.error = "schema_version mismatch: descriptor " + std::to_string(d.schema_version) +
              " vs header " + std::to_string(kFeatureSchemaVersion);
    return d;
  }
  if (static_cast<int>(d.features.size()) != kNumFeatures) {
    d.error = "feature count " + std::to_string(d.features.size()) + " != " +
              std::to_string(kNumFeatures);
    return d;
  }
  if (static_cast<int>(d.outputs.size()) != kNumOutputs) {
    d.error = "output count " + std::to_string(d.outputs.size()) + " != " +
              std::to_string(kNumOutputs);
    return d;
  }
  for (int k = 0; k < kNumFeatures; ++k) {
    if (d.features[k] != kFeatureNames[k]) {
      d.error = "feature order mismatch at " + std::to_string(k);
      return d;
    }
  }
  for (int k = 0; k < kNumOutputs; ++k) {
    if (d.outputs[k] != kOutputNames[k]) {
      d.error = "output order mismatch at " + std::to_string(k);
      return d;
    }
  }
  if (static_cast<int>(d.norm_mean.size()) != kNumFeatures ||
      static_cast<int>(d.norm_std.size()) != kNumFeatures) {
    d.error = "norm arrays are not " + std::to_string(kNumFeatures) + " long";
    return d;
  }
  for (const double v : d.norm_std) {
    if (!std::isfinite(v)) {
      d.error = "norm std is not finite";
      return d;
    }
  }
  d.valid = true;
  return d;
}

ModelDescriptor load_descriptor(const std::string& path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    ModelDescriptor d;
    d.error = "cannot open " + path;
    return d;
  }
  std::stringstream ss;
  ss << in.rdbuf();
  return parse_descriptor(ss.str());
}

// Declared in inference.hpp, defined in inference.cpp, which needs yaml-cpp.
FeatureNorm ModelDescriptor::norm() const {
  FeatureNorm n;
  for (int k = 0; k < kNumFeatures; ++k) {
    n.mean[k] = (k < static_cast<int>(norm_mean.size())) ? norm_mean[k] : 0.0;
    const double s = (k < static_cast<int>(norm_std.size())) ? norm_std[k] : 1.0;
    n.inv_std[k] = (std::fabs(s) > 1e-12) ? (1.0 / s) : 0.0;
  }
  return n;
}

LinearBackend::LinearBackend(int n_features, int n_outputs)
    : n_in_(n_features),
      n_out_(n_outputs),
      w_(static_cast<size_t>(n_features) * static_cast<size_t>(n_outputs), 0.0),
      b_(static_cast<size_t>(n_outputs), 0.0),
      y_(static_cast<size_t>(n_outputs), 0.0) {}

void LinearBackend::setWeights(std::vector<double> w, std::vector<double> b) {
  if (w.size() != w_.size() || b.size() != b_.size()) {
    error_ = "weight vector has the wrong size";
    ready_ = false;
    return;
  }
  w_ = std::move(w);
  b_ = std::move(b);
  for (const double v : w_) {
    if (!std::isfinite(v)) {
      error_ = "weights are not finite";
      ready_ = false;
      return;
    }
  }
  ready_ = true;
  error_.clear();
}

bool LinearBackend::run(const double* features, int n_features, double* outputs,
                        int n_outputs) {
  if (!ready_ || n_features != n_in_ || n_outputs != n_out_) {
    error_ = "run() called with the wrong shape or an unready backend";
    return false;
  }
  for (int o = 0; o < n_out_; ++o) {
    double acc = b_[static_cast<size_t>(o)];
    for (int f = 0; f < n_in_; ++f) {
      acc += w_[static_cast<size_t>(o) * n_in_ + f] * features[f];
    }
    y_[static_cast<size_t>(o)] = acc;
    outputs[o] = acc;
  }
  return true;
}

std::unique_ptr<IInferenceBackend> make_backend(const ModelDescriptor& desc,
                                                const std::string& dir) {
  if (!desc.valid) return nullptr;
  if (desc.backend != "linear") return nullptr;

  const std::string bin = dir + "/" + desc.weights_file;
  std::ifstream in(bin, std::ios::binary);
  if (!in.is_open()) return nullptr;
  double raw[3 * kNumFeatures + kNumOutputs];
  in.read(reinterpret_cast<char*>(raw), sizeof(raw));
  if (in.gcount() != static_cast<std::streamsize>(sizeof(raw))) return nullptr;

  auto backend = std::make_unique<LinearBackend>(kNumFeatures, kNumOutputs);
  std::vector<double> w(raw, raw + 3 * kNumFeatures);
  std::vector<double> b(raw + 3 * kNumFeatures, raw + 3 * kNumFeatures + kNumOutputs);
  backend->setWeights(std::move(w), std::move(b));
  return backend->ready() ? std::unique_ptr<IInferenceBackend>(backend.release()) : nullptr;
}

bool onnx_backend_available() { return false; }

}  // namespace tram
