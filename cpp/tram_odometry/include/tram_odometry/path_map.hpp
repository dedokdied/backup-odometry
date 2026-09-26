// Polyline route map ("pathgraph"): arc-length parameterisation, heading,
// curvature and grade, plus a fast nearest-point projection.
//
// The map is a plain CSV/TSV file (x y z [s]) in UTM metres, produced by
// tools/build_path_map.py from the train bags or from the organiser-provided
// route/elevation map. Keeping the format trivial means we are not blocked
// while the official map format is being specified.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace tram {

struct PathPoint {
  double x = 0.0;         ///< UTM easting, m
  double y = 0.0;         ///< UTM northing, m
  double z = 0.0;         ///< altitude, m
  double s = 0.0;         ///< arc length from the start of the map, m
  double heading = 0.0;   ///< rad, direction of travel
  double curvature = 0.0; ///< 1/m, positive = left turn
  double grade = 0.0;     ///< rad, path inclination
};

class PathMap {
 public:
  bool empty() const { return pts_.empty(); }
  size_t size() const { return pts_.size(); }
  const std::vector<PathPoint>& points() const { return pts_; }
  std::vector<PathPoint>& points() { return pts_; }

  /// Loads "x y z" or "x y z s" per line; '#' starts a comment.
  bool loadCsv(const std::string& path);

  /// Recomputes s/heading/curvature/grade from the geometry (call after load).
  void finalise();

  /// Nearest point projection. Returns false if the map is empty or the
  /// projection is further away than max_radius (when > 0).
  bool project(double x, double y, PathPoint& out, double& along, double& cross,
               double max_radius = 0.0) const;

  /// Interpolated point at arc length s (clamped to the map ends).
  bool pointAt(double s, PathPoint& out) const;

  double totalLength() const { return pts_.empty() ? 0.0 : pts_.back().s; }

  void clear();

 private:
  void buildIndex();
  int cellOf(double x, double y) const;

  std::vector<PathPoint> pts_;
  // Uniform grid (CSR-style) over the map bounding box for nearest search.
  double cell_ = 25.0;
  double min_x_ = 0.0, min_y_ = 0.0;
  int nx_ = 0, ny_ = 0;
  std::vector<int> cell_start_;
  std::vector<int> cell_items_;
};

}  // namespace tram
