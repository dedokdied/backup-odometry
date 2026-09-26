// Standalone extension of TEST(Estimator, EndToEndSyntheticRun) to full route
// length. Same scenario, same 3% wheel scale error, same 2 s of GNSS, but run
// for the whole 4.7 km instead of 160 m, and sampling s along the way so the
// growth of the drift can be classified as linear or exponential.
//
// Build: see run_blind_drift.sh

#include <cmath>
#include <cstdio>
#include <vector>

#include "tram_odometry/estimator.hpp"
#include "tram_odometry/params.hpp"
#include "tram_odometry/types.hpp"

using namespace tram;

int main() {
  Params p;
  p.rates.output_hz = 50.0;
  p.frame.origin_lat = 55.75;
  p.frame.origin_lon = 37.61;
  p.frame.origin_alt = 100.0;
  p.vehicle.mass_kg = 38000.0;
  p.path_map.enable = false;
  Estimator est(p);

  const double v_true = 8.0;
  const double wheel_kmh = v_true * 3.6 * 1.03;
  const double dt = 0.02;
  const int gnss_steps = 100;  // 2 s, as in the original test

  // 4708 m of route at 8 m/s.
  const int total_steps = static_cast<int>(4708.0 / v_true / dt);

  double t = 100.0;
  est.onGnssFix(t, 55.75, 37.61, 100.0, 12, 1, 0.01);
  est.onGnssVel(t, v_true, 0.0, 0.0);
  est.onDriver(t, 0.0);

  struct Sample {
    double s_true, s_est, err, ratio;
  };
  std::vector<Sample> samples;
  const int every = total_steps / 12;  // ~12 checkpoints

  bool ok = true;
  for (int i = 0; i < total_steps; ++i) {
    t += dt;
    est.onWheelFront(t, wheel_kmh);
    est.onWheelRear(t, wheel_kmh);
    est.onDriver(t, 0.0);
    if (i < gnss_steps) {
      est.onGnssFix(t, 55.75, 37.61, 100.0, 12, 1, 0.01);
      est.onGnssVel(t, v_true, 0.0, 0.0);
    }
    ok = est.step(t) && ok;

    if (every > 0 && (i + 1) % every == 0) {
      const Estimate e = est.estimate();
      const double s_true = (i + 1) * dt * v_true;
      const double err = e.s - s_true;
      samples.push_back({s_true, e.s, err, s_true > 0 ? err / s_true : 0.0});
    }
  }

  const Estimate e = est.estimate();
  const double s_true = total_steps * dt * v_true;
  const double err = e.s - s_true;
  const double drift_pct = 100.0 * std::fabs(err) / s_true;

  std::printf("steps=%d  duration=%.1f s  true s=%.1f m  est s=%.1f m\n",
              total_steps, total_steps * dt, s_true, e.s);
  std::printf("v=%.4f (true %.2f)  b_scale=%.6f (true %.6f)  trust=%.3f\n", e.v,
              v_true, e.b_scale, 1.0 / 1.03, e.trust);
  std::printf("FINAL DRIFT: %+.2f m  = %.3f %%  of %.0f m\n", err, drift_pct,
              s_true);
  std::printf("step() returned true throughout: %s\n", ok ? "yes" : "NO");

  std::printf("\n  s_true      err        err/s_true\n");
  for (const Sample& sm : samples) {
    std::printf("  %8.0f  %+8.2f   %+.5f\n", sm.s_true, sm.err, sm.ratio);
  }

  // Linear growth => err/s_true roughly constant. Exponential => ratio climbs.
  double r0 = 0.0, r1 = 0.0;
  if (samples.size() >= 2) {
    r0 = std::fabs(samples.front().ratio);
    r1 = std::fabs(samples.back().ratio);
  }
  std::printf("\nrelative drift first checkpoint %.5f, last %.5f, growth x%.2f\n",
              r0, r1, r0 > 1e-9 ? r1 / r0 : 0.0);
  std::printf("VERDICT drift_vs_2pct: %s\n", drift_pct < 2.0 ? "PASS" : "FAIL");

  return 0;
}
