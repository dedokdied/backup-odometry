// Drives Estimator over a dumped run and measures position drift.
//
// Two scenarios, because the difference between them is the only way to tell
// dead reckoning from map assistance apart:
//   path_map off - GNSS during the 2.5 s alignment window only, then blind
//   path_map on  - same, plus the route CSV snapping the position to the map
//
// The reference path comes from the bag's GNSS and is used only to score the
// run. It is never fed to the estimator beyond the alignment window.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "tram_odometry/estimator.hpp"
#include "tram_odometry/geo.hpp"
#include "tram_odometry/params.hpp"
#include "tram_odometry/types.hpp"

using namespace tram;

namespace {

struct Series {
  std::vector<double> t, v, u, omega_f, omega_r;
  std::vector<double> ref_t, ref_lat, ref_lon;
};

bool load_features(const std::string& path, Series& s) {
  std::ifstream in(path);
  if (!in.is_open()) return false;
  std::string line;
  if (!std::getline(in, line)) return false;
  int iT = -1, iV = -1, iU = -1, iOF = -1, iOR = -1;
  {
    std::stringstream hs(line);
    std::string c;
    int idx = 0;
    while (std::getline(hs, c, ',')) {
      if (c == "t") iT = idx;
      if (c == "v") iV = idx;
      if (c == "u") iU = idx;
      if (c == "omega_front") iOF = idx;
      if (c == "omega_rear") iOR = idx;
      ++idx;
    }
  }
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::vector<double> v;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) v.push_back(cell.empty() ? std::nan("") : std::atof(cell.c_str()));
    if (static_cast<int>(v.size()) <= iOR) continue;
    s.t.push_back(v[iT]);
    s.v.push_back(v[iV]);
    s.u.push_back(v[iU]);
    s.omega_f.push_back(v[iOF]);
    s.omega_r.push_back(v[iOR]);
  }
  return !s.t.empty();
}

void load_reference(const std::string& path, Series& s) {
  std::ifstream in(path);
  if (!in.is_open()) return;
  std::string line;
  if (!std::getline(in, line)) return;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::vector<double> v;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) v.push_back(cell.empty() ? std::nan("") : std::atof(cell.c_str()));
    if (v.size() < 4) continue;
    if (!std::isfinite(v[1]) || !std::isfinite(v[2])) continue;
    s.ref_t.push_back(v[0]);
    s.ref_lat.push_back(v[1]);
    s.ref_lon.push_back(v[2]);
  }
}

double gnss_path_length(const Series& s) {
  double total = 0.0;
  for (size_t i = 1; i < s.ref_lat.size(); ++i) {
    const double la1 = s.ref_lat[i - 1] * M_PI / 180.0, la2 = s.ref_lat[i] * M_PI / 180.0;
    const double x1 = s.ref_lon[i - 1] * M_PI / 180.0 * 6371000.0 * std::cos(la1);
    const double x2 = s.ref_lon[i] * M_PI / 180.0 * 6371000.0 * std::cos(la2);
    const double y1 = la1 * 6371000.0, y2 = la2 * 6371000.0;
    total += std::hypot(x2 - x1, y2 - y1);
  }
  return total;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::printf("usage: %s <features.csv> <reference.csv> <route_map.csv|-> <0|1 map>\n", argv[0]);
    return 2;
  }
  const bool use_map = std::atoi(argv[4]) != 0;
  const double ALIGN_S = 2.5;

  Series s;
  if (!load_features(argv[1], s)) {
    std::printf("cannot read features %s\n", argv[1]);
    return 1;
  }
  load_reference(argv[2], s);
  if (s.ref_lat.empty()) {
    std::printf("reference has no GNSS\n");
    return 1;
  }

  Params p;
  p.frame.origin_lat = s.ref_lat.front();
  p.frame.origin_lon = s.ref_lon.front();
  p.frame.origin_alt = 100.0;
  p.path_map.enable = use_map;
  if (use_map && std::string(argv[3]) != "-") {
    p.path_map.file = argv[3];
  }
  Estimator est(p);

  const double t0 = s.t.front();
  double sum_abs_s_err = 0.0;
  int n_s_err = 0;
  double max_s_err = 0.0;
  size_t ref_idx = 0;
  int gnss_fed = 0;

  for (size_t i = 0; i < s.t.size(); ++i) {
    const double t = t0 + (s.t[i] - t0);
    est.onWheelFront(t, s.v[i] * 3.6);
    est.onWheelRear(t, s.v[i] * 3.6);
    est.onDriver(t, s.u[i]);
    if (t - t0 < ALIGN_S) {
      while (ref_idx < s.ref_lat.size() && s.ref_t[ref_idx] <= t) ++ref_idx;
      const size_t k = (ref_idx == 0) ? 0 : ref_idx - 1;
      est.onGnssFix(t, s.ref_lat[k], s.ref_lon[k], 100.0, 12, 1, 0.01);
      est.onGnssVel(t, s.v[i], 0.0, 0.0);
      ++gnss_fed;
    }
    est.step(t);

  }

  const Estimate e = est.estimate();
  const double ref_len = gnss_path_length(s);
  const double duration = s.t.back() - t0;
  std::printf("scenario: path_map %s\n", use_map ? "ON" : "OFF");
  std::printf("  gnss samples fed (alignment only): %d\n", gnss_fed);
  std::printf("  duration %.1f s | estimator along-path s = %.1f m\n", duration, e.s);
  std::printf("  reference path length from GNSS      = %.1f m\n", ref_len);
  if (ref_len > 1.0) {
    std::printf("  along-path error                     = %+.1f m (%.3f %%)\n", e.s - ref_len,
                100.0 * (e.s - ref_len) / ref_len);
  }
  std::printf("  v = %.4f m/s | trust = %.3f | b_scale = %.5f | mu = %.4f\n", e.v, e.trust, e.b_scale, e.mu);
  std::printf("  x = %.2f  y = %.2f  z = %.2f (map frame: %s)\n", e.x, e.y, e.z, e.model_only ? "local DR" : "map");
  (void)sum_abs_s_err;
  (void)n_s_err;
  (void)max_s_err;
  return 0;
}
