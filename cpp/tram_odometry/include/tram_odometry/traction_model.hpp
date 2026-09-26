// Non-linear traction/brake model: controller position -> wheel force.
//
//   tau_cmd(u) = tau_max * Phi(u) * P_lim(v)
//   F_wheel    = tau_cmd * i_g * eta / R,  limited by constant power
//   dtau/dt    = (tau_cmd - tau) / tau_a
//
// Phi(u) is a piecewise-linear table (the real notch characteristic is neither
// linear in the handle position nor constant in speed).
#pragma once

#include "tram_odometry/params.hpp"
#include "tram_odometry/signal_filter.hpp"

namespace tram {

class TractionModel {
 public:
  TractionModel(const TractionParams& tp, const VehicleParams& vp);

  void reset();

  /// Advances the drive train. u is the normalised controller position,
  /// v the current body speed in m/s. Returns the commanded wheel force (N)
  /// before adhesion limiting.
  double step(double u, double v, double dt);

  /// Normalised non-linear shape of the controller characteristic, [0, 1].
  double shapeFactor(double u) const;

  double shaftTorque() const { return torque_; }
  double driveForce() const { return drive_force_; }
  double brakeForce() const { return brake_force_; }
  double powerLimit(double v) const;

 private:
  TractionParams tp_;
  VehicleParams vp_;
  FirstOrderLag drive_lag_;
  FirstOrderLag brake_lag_;
  double torque_ = 0.0;
  double drive_force_ = 0.0;
  double brake_force_ = 0.0;
};

}  // namespace tram
