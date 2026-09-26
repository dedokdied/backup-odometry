#include "tram_odometry/path_map.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace tram {
namespace {
constexpr double kTwoPi = 2.0 * 3.14159265358979323846;
inline double wrapPi(double a) {
  while (a > M_PI) a -= kTwoPi;
  while (a < -M_PI) a += kTwoPi;
  return a;
}
}  // namespace

void PathMap::clear() {
  pts_.clear();
  cell_start_.clear();
  cell_items_.clear();
  nx_ = ny_ = 0;
}

bool PathMap::loadCsv(const std::string& path) {
  std::ifstream in(path);
  if (!in.is_open()) return false;
  clear();
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#' || line[0] == '%') continue;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::replace(line.begin(), line.end(), '\t', ' ');
    std::istringstream ss(line);
    PathPoint p;
    if (!(ss >> p.x >> p.y)) continue;
    if (!(ss >> p.z)) p.z = 0.0;
    if (!(ss >> p.s)) p.s = 0.0;
    if (!std::isfinite(p.x) || !std::isfinite(p.y)) continue;
    pts_.push_back(p);
  }
  if (pts_.empty()) return false;
  finalise();
  return true;
}

void PathMap::finalise() {
  const size_t n = pts_.size();
  if (n == 0) return;

  // Arc length.
  pts_[0].s = 0.0;
  for (size_t i = 1; i < n; ++i) {
    const double dx = pts_[i].x - pts_[i - 1].x;
    const double dy = pts_[i].y - pts_[i - 1].y;
    pts_[i].s = pts_[i - 1].s + std::sqrt(dx * dx + dy * dy);
  }

  // Heading by central differences (endpoints one-sided).
  for (size_t i = 0; i < n; ++i) {
    const size_t a = (i == 0) ? 0 : i - 1;
    const size_t b = (i + 1 == n) ? n - 1 : i + 1;
    const double dx = pts_[b].x - pts_[a].x;
    const double dy = pts_[b].y - pts_[a].y;
    pts_[i].heading = std::atan2(dy, dx);
  }

  // Curvature from three consecutive points (Menger curvature), smoothed once.
  std::vector<double> curv(n, 0.0);
  for (size_t i = 1; i + 1 < n; ++i) {
    const double ax = pts_[i].x - pts_[i - 1].x, ay = pts_[i].y - pts_[i - 1].y;
    const double bx = pts_[i + 1].x - pts_[i].x, by = pts_[i + 1].y - pts_[i].y;
    const double la = std::sqrt(ax * ax + ay * ay), lb = std::sqrt(bx * bx + by * by);
    const double cross = ax * by - ay * bx;
    if (la > 1e-6 && lb > 1e-6) {
      curv[i] = 2.0 * cross / (la * lb * (la + lb));
    }
  }
  for (size_t i = 0; i < n; ++i) {
    const size_t a = (i == 0) ? 0 : i - 1;
    const size_t b = (i + 1 == n) ? n - 1 : i + 1;
    pts_[i].curvature = 0.5 * (curv[a] + curv[b]);
  }

  // Grade from the elevation profile (smoothed over ~20 m).
  for (size_t i = 0; i < n; ++i) {
    const size_t a = (i == 0) ? 0 : i - 1;
    const size_t b = (i + 1 == n) ? n - 1 : i + 1;
    const double ds = pts_[b].s - pts_[a].s;
    pts_[i].grade = (ds > 1e-3) ? std::atan2(pts_[b].z - pts_[a].z, ds) : 0.0;
  }

  buildIndex();
}

int PathMap::cellOf(double x, double y) const {
  if (nx_ <= 0 || ny_ <= 0) return -1;
  int ix = static_cast<int>(std::floor((x - min_x_) / cell_));
  int iy = static_cast<int>(std::floor((y - min_y_) / cell_));
  ix = std::clamp(ix, 0, nx_ - 1);
  iy = std::clamp(iy, 0, ny_ - 1);
  return iy * nx_ + ix;
}

void PathMap::buildIndex() {
  cell_start_.clear();
  cell_items_.clear();
  if (pts_.empty()) return;

  double max_x = pts_[0].x, max_y = pts_[0].y;
  min_x_ = max_x;
  min_y_ = max_y;
  for (const auto& p : pts_) {
    min_x_ = std::min(min_x_, p.x);
    min_y_ = std::min(min_y_, p.y);
    max_x = std::max(max_x, p.x);
    max_y = std::max(max_y, p.y);
  }
  cell_ = 25.0;
  nx_ = std::max(1, static_cast<int>((max_x - min_x_) / cell_) + 1);
  ny_ = std::max(1, static_cast<int>((max_y - min_y_) / cell_) + 1);
  // Guard against pathological maps.
  while (static_cast<long long>(nx_) * ny_ > 400000LL) {
    cell_ *= 2.0;
    nx_ = std::max(1, static_cast<int>((max_x - min_x_) / cell_) + 1);
    ny_ = std::max(1, static_cast<int>((max_y - min_y_) / cell_) + 1);
  }

  const size_t ncell = static_cast<size_t>(nx_) * static_cast<size_t>(ny_);
  cell_start_.assign(ncell + 1, 0);
  for (const auto& p : pts_) {
    ++cell_start_[static_cast<size_t>(cellOf(p.x, p.y)) + 1];
  }
  for (size_t i = 1; i <= ncell; ++i) cell_start_[i] += cell_start_[i - 1];
  cell_items_.resize(pts_.size());
  std::vector<int> cursor(cell_start_.begin(), cell_start_.end() - 1);
  for (size_t i = 0; i < pts_.size(); ++i) {
    cell_items_[static_cast<size_t>(cursor[static_cast<size_t>(cellOf(pts_[i].x, pts_[i].y))]++)] =
        static_cast<int>(i);
  }
}

bool PathMap::project(double x, double y, PathPoint& out, double& along, double& cross,
                      double max_radius) const {
  const size_t n = pts_.size();
  if (n == 0) return false;
  if (n == 1) {
    out = pts_[0];
    along = 0.0;
    cross = std::hypot(x - out.x, y - out.y);
    return max_radius <= 0.0 || cross <= max_radius;
  }

  // Coarse search in the grid neighbourhood, then refine on the segment.
  size_t best = 0;
  double best_d2 = 1e300;
  const int c = cellOf(x, y);
  const int ring = 1;
  if (c >= 0) {
    const int cx = c % nx_, cy = c / nx_;
    for (int dy = -ring; dy <= ring; ++dy) {
      for (int dx = -ring; dx <= ring; ++dx) {
        const int ix = cx + dx, iy = cy + dy;
        if (ix < 0 || iy < 0 || ix >= nx_ || iy >= ny_) continue;
        const size_t cc = static_cast<size_t>(iy) * nx_ + ix;
        for (int k = cell_start_[cc]; k < cell_start_[cc + 1]; ++k) {
          const size_t i = static_cast<size_t>(cell_items_[k]);
          const double ddx = pts_[i].x - x, ddy = pts_[i].y - y;
          const double d2 = ddx * ddx + ddy * ddy;
          if (d2 < best_d2) {
            best_d2 = d2;
            best = i;
          }
        }
      }
    }
  }
  if (best_d2 > 1e299) {  // index unusable: linear fallback
    for (size_t i = 0; i < n; ++i) {
      const double ddx = pts_[i].x - x, ddy = pts_[i].y - y;
      const double d2 = ddx * ddx + ddy * ddy;
      if (d2 < best_d2) {
        best_d2 = d2;
        best = i;
      }
    }
  }

  // Refine against the two adjacent segments.
  double best_cross = 1e300, best_s = 0.0, best_heading = 0.0, best_grade = 0.0, best_curv = 0.0;
  double best_z = 0.0;
  auto consider = [&](size_t i, size_t j) {
    if (j >= n) return;
    const double ax = pts_[i].x, ay = pts_[i].y;
    const double bx = pts_[j].x, by = pts_[j].y;
    const double ex = bx - ax, ey = by - ay;
    const double len2 = ex * ex + ey * ey;
    if (len2 < 1e-12) return;
    double t = ((x - ax) * ex + (y - ay) * ey) / len2;
    t = std::clamp(t, 0.0, 1.0);
    const double px = ax + t * ex, py = ay + t * ey;
    const double d2 = (x - px) * (x - px) + (y - py) * (y - py);
    if (d2 < best_cross) {
      best_cross = d2;
      best_s = pts_[i].s + t * std::sqrt(len2);
      best_heading = pts_[i].heading + t * wrapPi(pts_[j].heading - pts_[i].heading);
      best_grade = pts_[i].grade + t * (pts_[j].grade - pts_[i].grade);
      best_curv = pts_[i].curvature + t * (pts_[j].curvature - pts_[i].curvature);
      best_z = pts_[i].z + t * (pts_[j].z - pts_[i].z);
    }
  };
  if (best > 0) consider(best - 1, best);
  consider(best, best + 1);

  if (best_cross > 1e299) return false;
  best_cross = std::sqrt(best_cross);

  out.x = x;
  out.y = y;
  out.z = best_z;
  out.s = best_s;
  out.heading = best_heading;
  out.curvature = best_curv;
  out.grade = best_grade;
  along = best_s;
  cross = best_cross;
  return max_radius <= 0.0 || best_cross <= max_radius;
}

bool PathMap::pointAt(double s, PathPoint& out) const {
  const size_t n = pts_.size();
  if (n == 0) return false;
  if (n == 1 || s <= pts_.front().s) {
    out = pts_.front();
    return true;
  }
  if (s >= pts_.back().s) {
    out = pts_.back();
    return true;
  }
  // Binary search on the monotone s array.
  size_t lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const size_t mid = (lo + hi) / 2;
    if (pts_[mid].s <= s) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  const double ds = pts_[hi].s - pts_[lo].s;
  const double t = (ds > 1e-9) ? (s - pts_[lo].s) / ds : 0.0;
  out.x = pts_[lo].x + t * (pts_[hi].x - pts_[lo].x);
  out.y = pts_[lo].y + t * (pts_[hi].y - pts_[lo].y);
  out.z = pts_[lo].z + t * (pts_[hi].z - pts_[lo].z);
  out.s = s;
  out.heading = pts_[lo].heading + t * wrapPi(pts_[hi].heading - pts_[lo].heading);
  out.curvature = pts_[lo].curvature + t * (pts_[hi].curvature - pts_[lo].curvature);
  out.grade = pts_[lo].grade + t * (pts_[hi].grade - pts_[lo].grade);
  return true;
}

}  // namespace tram
