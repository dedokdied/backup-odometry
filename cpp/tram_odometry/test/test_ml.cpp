// Unit tests for the learned residual corrector: the schema contract, the
// built-in backends, the descriptor validation and the watchdog behaviour that
// guarantees a bad model can never take the odometry down.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "tram_odometry/inference.hpp"
#include "tram_odometry/longitudinal_model.hpp"
#include "tram_odometry/ml_corrector.hpp"
#include "tram_odometry/ml_features.hpp"

namespace {

using namespace tram;  // NOLINT(build/namespaces)

std::string write_file(const std::string& dir, const std::string& name,
                       const std::vector<double>& data) {
  std::filesystem::create_directories(dir);
  const std::string path = dir + "/" + name;
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(data.data()),
          static_cast<std::streamsize>(data.size() * sizeof(double)));
  return path;
}

std::string write_text(const std::string& dir, const std::string& name,
                       const std::string& text) {
  std::filesystem::create_directories(dir);
  const std::string path = dir + "/" + name;
  std::ofstream f(path);
  f << text;
  return path;
}

std::string descriptor_yaml(const std::string& backend, const std::string& weights,
                            const std::string& extra = "") {
  std::string feats;
  for (int i = 0; i < kNumFeatures; ++i) {
    feats += (i ? ", " : "") + std::string("\"") + kFeatureNames[i] + "\"";
  }
  std::string outs;
  for (int i = 0; i < kNumOutputs; ++i) {
    outs += (i ? ", " : "") + std::string("\"") + kOutputNames[i] + "\"";
  }
  return "schema_version: 1\n"
         "model_id: unit_test\n"
         "backend: \"" + backend + "\"\n"
         "weights_file: \"" + weights + "\"\n" +
         extra + "features: [" + feats + "]\n"
         "outputs: [" + outs + "]\n";
}

CorrectorInput make_input() {
  CorrectorInput in;
  in.t = 1.0;
  in.u = 0.5;
  in.v = 8.0;
  in.a_model = 0.4;
  in.grade = 0.01;
  in.mu = 0.35;
  in.omega_front = 26.0;
  in.omega_rear = 26.0;
  in.b_scale = 1.0;
  in.slip_index = 0.0;
  in.trust = 0.9;
  in.cmd_rate = 0.0;
  in.dt = 0.02;
  in.drive_force = 40000.0;
  in.brake_force = 0.0;
  in.adhesion_limit = 32000.0;
  return in;
}

}  // namespace

// --------------------------------------------------------------- schema

TEST(Features, NamesMatchIndices) {
  ASSERT_EQ(static_cast<int>(sizeof(kFeatureNames) / sizeof(kFeatureNames[0])),
            kNumFeatures);
  EXPECT_STREQ(kFeatureNames[kFControllerPos], "u");
  EXPECT_STREQ(kFeatureNames[kFSpeed], "v");
  EXPECT_STREQ(kFeatureNames[kFAccelModel], "a_model");
  EXPECT_STREQ(kFeatureNames[kFForceRatio], "force_ratio");
  EXPECT_STREQ(kOutputNames[kOAresidual], "a_residual");
  EXPECT_STREQ(kOutputNames[kOLogScale], "log_scale");
  EXPECT_STREQ(kOutputNames[kOMu], "mu");
}

TEST(Features, BuildIsFiniteAndDeterministic) {
  const CorrectorInput in = make_input();
  FeatureVector a, b;
  MlCorrector::buildFeatures(in, &a);
  MlCorrector::buildFeatures(in, &b);
  EXPECT_TRUE(a.finite());
  for (int i = 0; i < kNumFeatures; ++i) EXPECT_DOUBLE_EQ(a[i], b[i]);
  EXPECT_DOUBLE_EQ(a[kFUSq], 0.25);
  EXPECT_DOUBLE_EQ(a[kFAbsU], 0.5);
  EXPECT_NEAR(a[kFForceRatio], 40000.0 / 40000.0, 1e-9);
}

TEST(Features, NormalisationSurvivesDegenerateStd) {
  FeatureNorm n;
  for (int i = 0; i < kNumFeatures; ++i) {
    n.mean[i] = 1.0;
    n.inv_std[i] = 1.0 / 0.0;  // deliberately broken
  }
  FeatureVector in, out;
  in[kFSpeed] = 5.0;
  n.apply(in, out);
  EXPECT_TRUE(out.finite());
}

// -------------------------------------------------------------- backends

TEST(LinearBackend, ComputesAffineOutput) {
  LinearBackend b(kNumFeatures, kNumOutputs);
  std::vector<double> w(static_cast<size_t>(kNumFeatures) * kNumOutputs, 0.0);
  std::vector<double> bias(kNumOutputs, 0.0);
  w[static_cast<size_t>(kOAresidual) * kNumFeatures + kFSpeed] = 2.0;
  bias[kOAresidual] = -1.0;
  bias[kOMu] = 0.3;
  b.setWeights(w, bias);
  ASSERT_TRUE(b.ready());

  double in[kNumFeatures] = {};
  in[kFSpeed] = 3.0;
  double out[kNumOutputs] = {};
  ASSERT_TRUE(b.run(in, kNumFeatures, out, kNumOutputs));
  EXPECT_NEAR(out[kOAresidual], 5.0, 1e-12);
  EXPECT_NEAR(out[kOMu], 0.3, 1e-12);
}

TEST(MlpBackend, MatchesHandComputedTwoLayerNet) {
  // layers [in=2, hidden=1, out=1] on a reduced problem is not expressible
  // through the fixed schema, so exercise the evaluator with the real sizes and
  // verify it against an independent scalar implementation.
  MlpBackend b({kNumFeatures, 4, kNumOutputs});
  std::vector<double> w0(static_cast<size_t>(kNumFeatures) * 4, 0.0);
  std::vector<double> b0(4, 0.0);
  std::vector<double> w1(4 * kNumOutputs, 0.0);
  std::vector<double> b1(kNumOutputs, 0.0);
  for (int j = 0; j < 4; ++j) {
    w0[static_cast<size_t>(j) * kNumFeatures + kFSpeed] = 0.5 * (j + 1);
  }
  w1[0] = 1.0;
  b1[kOAresidual] = 0.25;
  std::vector<double> flat;
  flat.insert(flat.end(), w0.begin(), w0.end());
  flat.insert(flat.end(), b0.begin(), b0.end());
  flat.insert(flat.end(), w1.begin(), w1.end());
  flat.insert(flat.end(), b1.begin(), b1.end());
  b.setWeights(flat, flat.size());
  ASSERT_TRUE(b.ready());

  double in[kNumFeatures] = {};
  in[kFSpeed] = 2.0;
  double out[kNumOutputs] = {};
  ASSERT_TRUE(b.run(in, kNumFeatures, out, kNumOutputs));

  // Independent reference: hidden_j = tanh(w0[j][speed] * v + b0[j]),
  // out_0 = sum_i W1[0][i] * hidden_i + b1[0].
  double hidden[4];
  for (int j = 0; j < 4; ++j) {
    hidden[j] = std::tanh(w0[static_cast<size_t>(j) * kNumFeatures + kFSpeed] * in[kFSpeed] +
                          b0[j]);
  }
  double ref = b1[kOAresidual];
  for (int i = 0; i < 4; ++i) ref += w1[static_cast<size_t>(kOAresidual) * 4 + i] * hidden[i];
  EXPECT_NEAR(out[kOAresidual], ref, 1e-12);
  EXPECT_NEAR(out[kOAresidual], std::tanh(1.0) + 0.25, 1e-12);
}

TEST(MlpBackend, RejectsWrongWeightCount) {
  MlpBackend b({kNumFeatures, 4, kNumOutputs});
  b.setWeights(std::vector<double>(3, 0.0), 999);
  EXPECT_FALSE(b.ready());
}

// ------------------------------------------------------------ descriptor

TEST(Descriptor, RejectsWrongSchemaVersion) {
  std::string y = descriptor_yaml("null", "");
  y = "schema_version: 2\n" + y.substr(std::string("schema_version: 1\n").size());
  const ModelDescriptor d = parse_descriptor(y);
  EXPECT_FALSE(d.valid);
  EXPECT_NE(d.error.find("schema_version"), std::string::npos);
}

TEST(Descriptor, RejectsReorderedFeatures) {
  std::string y = descriptor_yaml("null", "");
  const std::string from = "features: [\"u\",";
  const size_t pos = y.find(from);
  ASSERT_NE(pos, std::string::npos);
  y.replace(pos, from.size(), "features: [\"v\",");
  const ModelDescriptor d = parse_descriptor(y);
  EXPECT_FALSE(d.valid);
  EXPECT_NE(d.error.find("feature 0"), std::string::npos);
}

TEST(Descriptor, RejectsMlpWithBadLayerSizes) {
  const ModelDescriptor d = parse_descriptor(
      descriptor_yaml("mlp", "w.bin", "layer_sizes: [4, 8, 3]\n"));
  EXPECT_FALSE(d.valid);
  EXPECT_NE(d.error.find("layer_sizes"), std::string::npos);
}

TEST(Descriptor, AcceptsNullBackend) {
  const ModelDescriptor d = parse_descriptor(descriptor_yaml("null", ""));
  EXPECT_TRUE(d.valid) << d.error;
}

TEST(Descriptor, MlpLayerSizeCountMatchesWeightCount) {
  const ModelDescriptor d = parse_descriptor(
      descriptor_yaml("mlp", "w.bin", "layer_sizes: [16, 8, 3]\n"));
  ASSERT_TRUE(d.valid) << d.error;
  size_t need = 0;
  for (size_t l = 0; l + 1 < d.layer_sizes.size(); ++l) {
    need += static_cast<size_t>(d.layer_sizes[l]) * static_cast<size_t>(d.layer_sizes[l + 1]) +
            static_cast<size_t>(d.layer_sizes[l + 1]);
  }
  EXPECT_EQ(need, static_cast<size_t>(kNumFeatures) * 8 + 8 + 8 * 3 + 3);
}

// -------------------------------------------------------------- corrector

TEST(Corrector, DisabledByDefault) {
  MlCorrector c;
  MlParams p;  // enable == false
  c.configure(p, "");
  EXPECT_FALSE(c.hasModel());
  const CorrectorOutput o = c.step(make_input());
  EXPECT_FALSE(o.applied);
  EXPECT_DOUBLE_EQ(o.a_residual, 0.0);
}

TEST(Corrector, AppliesLinearCorrectionAndClips) {
  const std::string dir = "tram_ml_test";
  // y0 = 5 m/s^2 on the first feature, which must be clipped to the limit.
  std::vector<double> w(static_cast<size_t>(kNumFeatures) * kNumOutputs, 0.0);
  std::vector<double> b(kNumOutputs, 0.0);
  w[kFControllerPos] = 100.0;
  w[static_cast<size_t>(kOLogScale) * kNumFeatures + kFControllerPos] = 100.0;
  std::vector<double> flat(w);
  flat.insert(flat.end(), b.begin(), b.end());
  write_file(dir, "w.bin", flat);
  write_text(dir, "m.yaml", descriptor_yaml("linear", "w.bin"));

  MlCorrector c;
  MlParams p;
  p.enable = true;
  p.model_dir = dir;
  p.descriptor = "m.yaml";
  c.configure(p, dir);
  ASSERT_TRUE(c.hasModel()) << c.statusLine();

  const CorrectorOutput o = c.step(make_input());
  EXPECT_TRUE(o.applied);
  // The clip must land exactly on the configured limit. Referencing the contract
  // constant instead of a literal keeps the test meaningful when the limit is
  // retuned, which it was: 0.80 -> 1.50 once the training labels were measured.
  EXPECT_NEAR(o.a_residual, std::min(std::abs(p.max_a_residual),
                                    kDefaultMaxAccelResidual), 1e-9);
  EXPECT_NEAR(o.log_scale, 0.05, 1e-9);
}

TEST(Corrector, RejectsNonFiniteState) {
  const std::string dir = "tram_ml_test";
  std::vector<double> flat(static_cast<size_t>(kNumFeatures) * kNumOutputs + kNumOutputs, 0.0);
  write_file(dir, "w.bin", flat);
  write_text(dir, "m.yaml", descriptor_yaml("linear", "w.bin"));

  MlCorrector c;
  MlParams p;
  p.enable = true;
  p.model_dir = dir;
  p.descriptor = "m.yaml";
  c.configure(p, dir);
  ASSERT_TRUE(c.hasModel());

  CorrectorInput bad = make_input();
  bad.v = std::nan("");
  const CorrectorOutput o = c.step(bad);
  EXPECT_FALSE(o.applied);
  EXPECT_GT(c.stats().rejected, 0u);
}

TEST(Corrector, MissingWeightsDegradeToPhysics) {
  const std::string dir = "tram_ml_test_bad";
  write_text(dir, "m.yaml", descriptor_yaml("linear", "does_not_exist.bin"));
  MlCorrector c;
  MlParams p;
  p.enable = true;
  p.model_dir = dir;
  p.descriptor = "m.yaml";
  c.configure(p, dir);
  EXPECT_FALSE(c.hasModel());
  const CorrectorOutput o = c.step(make_input());
  EXPECT_FALSE(o.applied);
  EXPECT_DOUBLE_EQ(o.a_residual, 0.0);
}

TEST(Corrector, CorruptWeightFileDegradesToPhysics) {
  const std::string dir = "tram_ml_test_corrupt";
  write_file(dir, "w.bin", std::vector<double>{1.0, 2.0});  // wrong length
  write_text(dir, "m.yaml", descriptor_yaml("linear", "w.bin"));
  MlCorrector c;
  MlParams p;
  p.enable = true;
  p.model_dir = dir;
  p.descriptor = "m.yaml";
  c.configure(p, dir);
  EXPECT_FALSE(c.hasModel());
}

TEST(Corrector, BlendZeroKeepsPhysics) {
  const std::string dir = "tram_ml_test";
  std::vector<double> flat(static_cast<size_t>(kNumFeatures) * kNumOutputs + kNumOutputs, 1.0);
  write_file(dir, "w2.bin", flat);
  write_text(dir, "m2.yaml", descriptor_yaml("linear", "w2.bin"));
  MlCorrector c;
  MlParams p;
  p.enable = true;
  p.blend = 0.0;
  p.model_dir = dir;
  p.descriptor = "m.yaml";
  c.configure(p, dir);
  ASSERT_TRUE(c.hasModel());
  const CorrectorOutput o = c.step(make_input());
  EXPECT_FALSE(o.applied);
}

TEST(Corrector, PhysicsUnchangedWhenResidualIsZero) {
  DynamicsParams dp;
  VehicleParams vp;
  AdhesionParams ap;
  LongitudinalModel m0(dp, vp, ap);
  LongitudinalModel m1(dp, vp, ap);
  LongitudinalModel::Input in;
  in.v = 10.0;
  in.drive_force = 50000.0;
  in.grade = 0.02;
  in.residual_accel = 0.0;
  const double a0 = m0.acceleration(in);
  const double a1 = m1.acceleration(in);
  EXPECT_DOUBLE_EQ(a0, a1);
  in.residual_accel = 0.1;
  EXPECT_NEAR(m1.acceleration(in) - a0, 0.1, 1e-12);
}
