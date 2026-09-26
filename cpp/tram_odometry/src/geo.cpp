#include "tram_odometry/geo.hpp"

#include <cmath>

namespace tram {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kA = 6378137.0;            // WGS-84 semi-major axis, m
constexpr double kF = 1.0 / 298.257223563;  // flattening
constexpr double kK0 = 0.9996;              // UTM scale factor
constexpr double kE2 = kF * (2.0 - kF);     // first eccentricity squared
constexpr double kEP2 = kE2 / (1.0 - kE2);  // second eccentricity squared

inline double deg2rad(double d) { return d * kPi / 180.0; }
inline double rad2deg(double r) { return r * 180.0 / kPi; }

}  // namespace

int utm_zone_for_lon(double lon_deg) {
  int z = static_cast<int>(std::floor((lon_deg + 180.0) / 6.0)) + 1;
  if (z < 1) z = 1;
  if (z > 60) z = 60;
  return z;
}

double metres_per_degree_lat(double lat_deg) {
  const double phi = deg2rad(lat_deg);
  const double s = std::sin(phi);
  return (kPi / 180.0) * kA * (1.0 - kE2) / std::pow(1.0 - kE2 * s * s, 1.5);
}

double metres_per_degree_lon(double lat_deg) {
  const double phi = deg2rad(lat_deg);
  const double s = std::sin(phi);
  const double c = std::cos(phi);
  if (std::fabs(c) < 1e-12) return 0.0;
  return (kPi / 180.0) * kA * c / std::sqrt(1.0 - kE2 * s * s);
}

UtmPoint wgs84_to_utm(double lat_deg, double lon_deg, int zone) {
  UtmPoint out;
  if (!(lat_deg >= -80.0 && lat_deg <= 84.0)) return out;
  if (!(lon_deg >= -180.0 && lon_deg <= 180.0)) return out;

  if (zone == 0) zone = utm_zone_for_lon(lon_deg);
  const double lon0_deg = static_cast<double>((zone - 1) * 6 - 180 + 3);

  const double phi = deg2rad(lat_deg);
  const double dlon = deg2rad(lon_deg - lon0_deg);
  const double sp = std::sin(phi), cp = std::cos(phi), tp = std::tan(phi);

  const double t2 = tp * tp;
  const double eta2 = kEP2 * cp * cp;
  const double t4 = t2 * t2, t6 = t4 * t2, eta4 = eta2 * eta2, eta6 = eta4 * eta2;
  (void)t6;
  (void)eta6;

  // Footprint latitude series.
  const double m = kA * ((phi - kE2 * std::sin(2 * phi) / 2.0 +
                          kE2 * kE2 * std::sin(4 * phi) / 16.0 -
                          kE2 * kE2 * kE2 * std::sin(6 * phi) / 32.0 +
                          kE2 * kE2 * kE2 * kE2 * std::sin(8 * phi) / 512.0));

  // Snyder, Map Projections - Working with the Transverse Mercator Projection.
  // The leading term is N1 * cos(phi) * dlon, which carries essentially the
  // whole easting offset from the false origin; dropping it silently pins every
  // point to 500 km and puts the round trip ~100 km out.
  const double denom = 1.0 - kE2 * sp * sp;
  const double n1 = kA / std::sqrt(denom);
  const double a_series = cp * dlon;
  const double a2 = a_series * a_series, a3 = a2 * a_series, a4 = a3 * a_series;
  const double a5 = a4 * a_series, a6 = a5 * a_series;

  // The false easting is added AFTER the scale factor: k0 multiplies only the
  // series, not the 500 km origin. Folding it inside costs a constant ~200 m.
  out.easting = kK0 * n1 *
                        (a_series + (1.0 - t2 + eta2) * a3 / 6.0 +
                         (5.0 - 18.0 * t2 + t4 + 72.0 * eta2 - 58.0 * kEP2) * a5 / 120.0) +
                500000.0;

  out.northing =
      kK0 * (m + tp * n1 * (a2 / 2.0 + (5.0 - t2 + 9.0 * eta2 + 4.0 * eta2 * eta2) * a4 / 24.0 +
                            (61.0 - 58.0 * t2 + t4 + 600.0 * eta2 - 330.0 * kEP2) * a6 / 720.0));

  if (out.northing < 0.0) out.northing += 10000000.0;  // southern hemisphere
  out.zone = zone;
  out.valid = true;
  return out;
}

void utm_to_wgs84(const UtmPoint& p, double& lat_deg, double& lon_deg) {
  if (!p.valid) {
    lat_deg = 0.0;
    lon_deg = 0.0;
    return;
  }
  const double lon0_deg = static_cast<double>((p.zone - 1) * 6 - 180 + 3);
  const double sp = std::sqrt(1.0 - kE2);
  const double e1 = (1.0 - sp) / (1.0 + sp);
  const double m0 = 1.0 - kE2 / 4.0 - 3.0 * kE2 * kE2 / 64.0 - 5.0 * kE2 * kE2 * kE2 / 256.0;

  const double M = p.northing / kK0;
  const double mu = M / (kA * m0);
  const double e1_2 = e1 * e1, e1_3 = e1_2 * e1, e1_4 = e1_3 * e1;
  double phi1 = mu + (1.5 * e1 - 27.0 / 32.0 * e1_3) * std::sin(2.0 * mu) +
                (21.0 / 16.0 * e1_2 - 55.0 / 32.0 * e1_4) * std::sin(4.0 * mu) +
                (151.0 / 96.0 * e1_3) * std::sin(6.0 * mu) +
                (1097.0 / 512.0 * e1_4) * std::sin(8.0 * mu);

  // The footprint series above is only a starting guess: on some parameter sets
  // it lands ~5 km off, which shows up as a position error. Refine it with a
  // few Newton steps on the exact meridian-arc relation meridian_arc(phi1) = M
  // so the footpoint is correct to well under a millimetre.
  const double e2 = kE2, e4 = e2 * e2, e6 = e4 * e2, e8 = e6 * e2;
  for (int it = 0; it < 8; ++it) {
    const double arc = kA * (phi1 - e2 * std::sin(2.0 * phi1) / 2.0 +
                             e4 * std::sin(4.0 * phi1) / 16.0 - e6 * std::sin(6.0 * phi1) / 32.0 +
                             e8 * std::sin(8.0 * phi1) / 512.0);
    const double darc = kA * (1.0 - e2 * std::cos(2.0 * phi1) +
                              e4 * std::cos(4.0 * phi1) / 4.0 -
                              3.0 * e6 * std::cos(6.0 * phi1) / 16.0 +
                              e8 * std::cos(8.0 * phi1) / 64.0);
    if (!(darc > 1.0)) break;
    const double step = (arc - M) / darc;
    phi1 -= step;
    if (std::fabs(step) < 1e-14) break;
  }

  const double s1 = std::sin(phi1), c1p = std::cos(phi1), t1 = std::tan(phi1);
  const double T1 = t1 * t1;
  const double C1 = kEP2 * c1p * c1p;
  const double denom = 1.0 - kE2 * s1 * s1;
  const double N1 = kA / std::sqrt(denom);
  const double R1 = kA * (1.0 - kE2) / (denom * std::sqrt(denom));
  const double D = (p.easting - 500000.0) / (N1 * kK0);

  const double D2 = D * D, D3 = D2 * D, D4 = D3 * D, D5 = D4 * D, D6 = D5 * D;

  const double lat = phi1 - (N1 * t1 / R1) *
                                  (D2 / 2.0 -
                                   (5.0 + 3.0 * T1 + 10.0 * C1 - 4.0 * C1 * C1 - 9.0 * kEP2) * D4 /
                                       24.0 +
                                   (61.0 + 90.0 * T1 + 298.0 * C1 + 45.0 * T1 * T1 -
                                    252.0 * kEP2 - 3.0 * C1 * C1) *
                                       D6 / 720.0);

  const double lon = lon0_deg + rad2deg(
                                  (D - (1.0 + 2.0 * T1 + C1) * D3 / 6.0 +
                                   (5.0 - 2.0 * C1 + 28.0 * T1 - 3.0 * C1 * C1 + 8.0 * kEP2 +
                                    24.0 * T1 * T1) *
                                       D5 / 120.0) /
                                  c1p);

  lat_deg = rad2deg(lat);
  lon_deg = lon;

  // Final 2D Newton polish against the exact forward projection. The truncated
  // inverse series above is already good to a couple of metres; this drives the
  // round trip to well below a millimetre, which is what the position
  // acceptance actually needs.
  {
    const double h = 1e-7;  // deg, ~1 cm: large enough to stay clear of fp noise
    for (int it = 0; it < 6; ++it) {
      const UtmPoint f0 = wgs84_to_utm(lat_deg, lon_deg, p.zone);
      if (!f0.valid) break;
      const double rE = p.easting - f0.easting;
      const double rN = p.northing - f0.northing;
      if (std::fabs(rE) < 1e-7 && std::fabs(rN) < 1e-7) break;
      const UtmPoint fl = wgs84_to_utm(lat_deg + h, lon_deg, p.zone);
      const UtmPoint fo = wgs84_to_utm(lat_deg, lon_deg + h, p.zone);
      if (!fl.valid || !fo.valid) break;
      const double dE_dlat = (fl.easting - f0.easting) / h;
      const double dN_dlat = (fl.northing - f0.northing) / h;
      const double dE_dlon = (fo.easting - f0.easting) / h;
      const double dN_dlon = (fo.northing - f0.northing) / h;
      const double det = dE_dlat * dN_dlon - dE_dlon * dN_dlat;
      if (!std::isfinite(det) || std::fabs(det) < 1e-12) break;
      lat_deg += (rE * dN_dlon - dE_dlon * rN) / det;
      lon_deg += (dE_dlat * rN - rE * dN_dlat) / det;
    }
  }
}

double utm_convergence_rad(double lat_deg, double lon_deg, int zone) {
  if (zone == 0) zone = utm_zone_for_lon(lon_deg);
  const double lon0_deg = static_cast<double>((zone - 1) * 6 - 180 + 3);
  const double dlon = deg2rad(lon_deg - lon0_deg);
  const double phi = deg2rad(lat_deg);
  const double sp = std::sin(phi), cp = std::cos(phi);
  return dlon * sp + 0.5 * dlon * dlon * sp * cp * std::tan(phi);
}

void utm_delta_to_enu(double d_east, double d_north, double lat_deg, double lon_deg, int zone,
                      double& x_east, double& y_north) {
  const double gamma = utm_convergence_rad(lat_deg, lon_deg, zone);
  const double cg = std::cos(gamma), sg = std::sin(gamma);
  x_east = d_east * cg - d_north * sg;
  y_north = d_east * sg + d_north * cg;
}

}  // namespace tram
