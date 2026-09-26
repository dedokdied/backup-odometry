// WGS-84 geodetic <-> UTM/MGRS conversions and local ENU helpers.
//
// The judging reference is given in MGRS (UTM) cartesian coordinates, so the
// node has to produce UTM easting/northing itself. Implemented with the
// classic USGS transverse-Mercator series: sub-millimetre inside a zone,
// which is far beyond the required accuracy.
#pragma once

#include <string>

namespace tram {

struct UtmPoint {
  double easting = 0.0;   ///< m
  double northing = 0.0;  ///< m
  int zone = 0;           ///< 1..60
  bool valid = false;
};

/// Standard 6-degree UTM zone for a longitude (floor((lon+180)/6)+1).
int utm_zone_for_lon(double lon_deg);

/// Forward projection. zone == 0 selects the zone automatically from lon.
UtmPoint wgs84_to_utm(double lat_deg, double lon_deg, int zone = 0);

/// Inverse projection (needed for map/route self-checks and tests).
void utm_to_wgs84(const UtmPoint& p, double& lat_deg, double& lon_deg);

/// Meridian convergence (gon) of a UTM zone at the given latitude.
double utm_convergence_rad(double lat_deg, double lon_deg, int zone);

/// Local REP-103 frame (x east, y north) anchored at (lat0, lon0).
/// Uses the UTM difference rotated by the meridian convergence, so the result
/// is a true tangent-plane ENU rather than a raw UTM delta.
void utm_delta_to_enu(double d_east, double d_north, double lat_deg, double lon_deg,
                      int zone, double& x_east, double& y_north);

/// Metres per degree (equatorial) helpers for quick local scaling.
double metres_per_degree_lat(double lat_deg);
double metres_per_degree_lon(double lat_deg);

}  // namespace tram
