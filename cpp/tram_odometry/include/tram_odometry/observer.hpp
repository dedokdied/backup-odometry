// Error-state Kalman filter (ESKF) for the fallback odometry.
//
//   x = [ v, a, s, b_scale, kappa_front, kappa_rear ]
//
//   v - longitudinal speed          (m/s)
//   a - acceleration                (m/s^2)
//   s - travelled distance          (m)
//   b_scale - odometry scale = R_eff / R_nom  (absorbs wheel wear, radius error)
//   kappa_f/r - longitudinal slip ratios of the two bogies
//
// Both bogies measure the *same* body velocity, so the measurement equation is
//
//   v = b_scale * R * omega_i * (1 - kappa_i),   i in {front, rear}
//
// i.e. the odometry is modelled as a velocity measurement whose gain and slip
// are unknown parameters that the filter estimates online. When the wheels slip,
// the confidence coming from the slip detector inflates R, which smoothly
// removes the odometry from the solution and leaves the model driving the state.
#pragma once

#include "tram_odometry/linalg.hpp"
#include "tram_odometry/params.hpp"

namespace tram {

class LongitudinalModel;

class Observer {
 public:
  static constexpr int kN = 6;

  Observer(const ObserverParams& op, const VehicleParams& vp);

  void reset();
  void setInitialVelocity(double v);
  void setInitialPosition(double s);

  /// Model-based prediction. dt in seconds.
  void predict(double dt, const LongitudinalModel& model, double drive_force,
               double brake_force, double grade, double curvature, double mu,
               double residual_accel = 0.0);

  /// Wheel odometry update. Returns false if the innovation was gated out.
  bool updateWheels(double omega_front, double omega_rear, double trust, double slip_index);

  /// GNSS velocity update (used during the initialisation window only).
  bool updateGnssVelocity(double v_meas);

  /// GNSS path-distance update, also performs the odometry scale calibration.
  bool updateGnssPath(double s_meas, double s_meas_cov);

  /// Explicit least-squares scale calibration from a reference distance.
  void calibrateScale(double s_odometry, double s_reference);

  /// Identify the wheel scale from a reference speed (GNSS). A wheel-speed sample
  /// alone cannot separate v from b, so this is what makes b observable.
  void calibrateScaleFromVelocity(double v_ref, double v_wheel);

  /// Slow multiplicative nudge of the odometry scale coming from the learned
  /// corrector. Relaxed rather than applied instantly so the filter covariance
  /// stays consistent, and P(b) is inflated to cover the correction.
  void applyScaleBias(double log_scale, double dt);

  double v() const { return x_[kV]; }
  double a() const { return x_[kA]; }
  double s() const { return x_[kS]; }
  double bScale() const { return x_[kB]; }
  double kappaFront() const { return x_[kKf]; }
  double kappaRear() const { return x_[kKr]; }
  double innovation() const { return last_innovation_; }
  double varV() const { return P_(kV, kV); }
  double varA() const { return P_(kA, kA); }
  double varS() const { return P_(kS, kS); }
  double varScale() const { return P_(kB, kB); }
  const Vec<kN>& state() const { return x_; }
  const Mat<kN>& covariance() const { return P_; }

 private:
  enum : int { kV = 0, kA = 1, kS = 2, kB = 3, kKf = 4, kKr = 5 };

  /// Generic scalar measurement update with an explicit row of H.
  bool scalarUpdate(int index, double innovation, double h_v, double h_a, double h_s,
                    double h_b, double h_kf, double h_kr, double r);

  ObserverParams op_;
  VehicleParams vp_;
  Vec<kN> x_{};
  Mat<kN> P_{};
  double last_innovation_ = 0.0;
};

}  // namespace tram
