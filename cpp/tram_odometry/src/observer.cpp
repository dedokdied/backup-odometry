#include "tram_odometry/observer.hpp"

#include <algorithm>
#include <cmath>

#include "tram_odometry/longitudinal_model.hpp"

namespace tram {

Observer::Observer(const ObserverParams& op, const VehicleParams& vp) : op_(op), vp_(vp) {
  reset();
}

void Observer::reset() {
  x_.setZero();
  x_[kV] = 0.0;
  x_[kA] = 0.0;
  x_[kS] = 0.0;
  x_[kB] = 1.0;
  x_[kKf] = 0.0;
  x_[kKr] = 0.0;

  P_.setZero();
  P_(kV, kV) = 25.0;       // speed is essentially unknown before the first fix
  P_(kA, kA) = 9.0;
  P_(kS, kS) = 100.0;
  P_(kB, kB) = 0.01;       // +-10 % scale prior
  P_(kKf, kKf) = 0.05;
  P_(kKr, kKr) = 0.05;
  last_innovation_ = 0.0;
}

void Observer::setInitialVelocity(double v) {
  x_[kV] = v;
  P_(kV, kV) = std::max(1e-3, op_.r_gnss_vel);
}

void Observer::setInitialPosition(double s) {
  x_[kS] = s;
  P_(kS, kS) = std::max(1e-3, op_.r_gnss_pos);
}

void Observer::predict(double dt, const LongitudinalModel& model, double drive_force,
                       double brake_force, double grade, double curvature, double mu,
                       double residual_accel) {
  if (dt <= 0.0 || dt > 1.0) return;

  LongitudinalModel::Input in;
  in.v = x_[kV];
  in.drive_force = drive_force;
  in.brake_force = brake_force;
  in.grade = grade;
  in.curvature = curvature;
  in.mu = mu;
  in.residual_accel = residual_accel;

  const double a_model = model.acceleration(in);
  const double v_prev = x_[kV];
  const double v_new = model.stepVelocity(in, v_prev, dt);

  // State propagation (trapezoidal integration of the distance).
  x_[kV] = v_new;
  x_[kA] = 0.5 * (x_[kA] + a_model);  // light smoothing of the model acceleration
  x_[kS] += 0.5 * (v_prev + x_[kV]) * dt;

  // Slip states relax towards the adhesion-limited slip implied by the model.
  const double f_limit = model.adhesionLimit(mu);
  const double kappa_target = model.slipFromExcess(std::max(drive_force, brake_force), f_limit);
  const double relax = std::min(1.0, dt / 0.2);
  x_[kKf] += (kappa_target - x_[kKf]) * relax;
  x_[kKr] += (kappa_target - x_[kKr]) * relax;

  // Jacobian of the discrete prediction.
  Mat<kN> F;
  F.setIdentity();
  F(kS, kV) = dt;
  F(kKf, kV) = -0.05 * dt;  // slip grows with speed
  F(kKr, kV) = -0.05 * dt;

  // Process noise.
  Mat<kN> Q;
  Q.setZero();
  Q(kV, kV) = op_.sigma_v * op_.sigma_v * dt;
  Q(kA, kA) = op_.sigma_a * op_.sigma_a * dt;
  Q(kS, kS) = op_.sigma_v * op_.sigma_v * dt;
  Q(kB, kB) = op_.sigma_scale * op_.sigma_scale * dt;
  Q(kKf, kKf) = op_.sigma_slip * op_.sigma_slip * dt;
  Q(kKr, kKr) = op_.sigma_slip * op_.sigma_slip * dt;

  P_ = F * P_ * transpose(F) + Q;
  symmetrize(P_);

  // Hard physical bounds: the filter must never produce nonsense.
  x_[kV] = std::clamp(x_[kV], -20.0, 30.0);
  x_[kB] = std::clamp(x_[kB], 0.80, 1.25);
  x_[kKf] = std::clamp(x_[kKf], -1.0, 1.0);
  x_[kKr] = std::clamp(x_[kKr], -1.0, 1.0);
}

bool Observer::scalarUpdate(int index, double innovation, double h_v, double h_a, double h_s,
                            double h_b, double h_kf, double h_kr, double r) {
  const double h[kN] = {h_v, h_a, h_s, h_b, h_kf, h_kr};

  // S = H P H^T + R  (scalar, 1x1)
  double hp[kN];
  for (int i = 0; i < kN; ++i) {
    double s = 0.0;
    for (int j = 0; j < kN; ++j) s += P_(i, j) * h[j];
    hp[i] = s;
  }
  double S = r;
  for (int i = 0; i < kN; ++i) S += h[i] * hp[i];
  if (!(S > 1e-12)) return false;

  // Physical sanity check applied *before* the chi-square gate. S contains the
  // whole covariance, so while P is large the gate opens up and NIS stays small
  // even for an absurd residual: the filter would believe anything because it
  // believes it knows nothing. A tram cannot change its speed by 10 m/s within
  // one 20 ms cycle (that would be 500 m/s^2), so bound the residual in absolute
  // terms. index 2 is the along-path GNSS distance, which lives in metres.
  const double max_innov = (index == 2) ? 50.0 : 10.0;
  if (std::abs(innovation) > max_innov) {
    last_innovation_ = innovation * innovation / S;
    return false;
  }

  const double nis = innovation * innovation / S;
  last_innovation_ = nis;
  if (nis > op_.gate_chi2) return false;  // outlier: reject, keep prediction

  for (int i = 0; i < kN; ++i) {
    const double k_i = hp[i] / S;
    x_[i] += k_i * innovation;
  }

  // Joseph form: numerically stable for long runs.
  // R is scalar here, so K R K^T = r * K K^T and K = H^T S^-1 row-wise,
  // i.e. KH(i,j) = h[i] * h[j] / S.
  Mat<kN> I;
  I.setIdentity();
  Mat<kN> KH;
  KH.setZero();
  for (int i = 0; i < kN; ++i)
    for (int j = 0; j < kN; ++j) KH(i, j) = h[i] * h[j] / S;
  const Mat<kN> A = I - KH;
  const Mat<kN> Kr = KH * r;
  P_ = A * P_ * transpose(A) + Kr * transpose(KH);
  symmetrize(P_);

  x_[kB] = std::clamp(x_[kB], 0.80, 1.25);
  (void)index;
  return true;
}

bool Observer::updateWheels(double omega_front, double omega_rear, double trust,
                            double slip_index) {
  const double R = vp_.wheel_radius_m;
  const double b = x_[kB];

  // The raw measurement is the wheel surface speed z = R*omega, and the predicted
  // measurement is h(x) = v / b.
  //
  // The slip states kf/kr are deliberately NOT part of h. They come from
  // slipFromExcess(), i.e. from the adhesion model, so putting them in the
  // measurement equation asserts that a heuristic is exactly right. With mu
  // mis-estimated the target slip came out around 0.2, and since h = v/(b(1-k))
  // that is a hard 20 % multiplicative underestimate of the speed, not a
  // confidence loss. Slip belongs in the measurement noise instead: the
  // `inflate` factor below widens R until the gain collapses and the filter
  // leans on the model, which is the honest representation of "the wheels are
  // unreliable" that does not fabricate a velocity.
  const double zf = R * omega_front;
  const double zr = R * omega_rear;

  const double eps = 1e-3;
  const double safe_b = std::copysign(std::max(std::abs(b), eps), b);
  const double hf = x_[kV] / safe_b;
  const double hr = x_[kV] / safe_b;
  const double yf = zf - hf;
  const double yr = zr - hr;

  // H = dh/dx, with dh/db held at zero on purpose. A single wheel sample only
  // constrains the product v = b*z, so leaving b free here makes the persistent
  // residual left by rolling resistance push b into its clamp and drag v with
  // it. b is identified separately from the GNSS velocity ratio instead.
  const double hv = 1.0 / safe_b;

  // Adaptive measurement noise: full trust -> tight, no trust -> odometry off.
  // This is where a slip indication is turned into loss of confidence.
  const double w = std::clamp(trust, 0.0, 1.0);
  const double base = op_.r_wheel * w + op_.r_wheel_degraded * (1.0 - w);
  const double inflate = 1.0 + 4.0 * slip_index;
  const double r = base * inflate;

  const bool ok_f = scalarUpdate(0, yf, hv, 0.0, 0.0, 0.0, 0.0, 0.0, r);
  const bool ok_r = scalarUpdate(1, yr, hv, 0.0, 0.0, 0.0, 0.0, 0.0, r);
  return ok_f || ok_r;
}

void Observer::calibrateScaleFromVelocity(double v_ref, double v_wheel) {
  // The wheel scale is the ratio between a reference speed and what the
  // tachometers report for it. Only called while a GNSS reference exists, which
  // is what makes b observable at all.
  if (!std::isfinite(v_ref) || !std::isfinite(v_wheel)) return;
  if (std::fabs(v_wheel) < 1.0) return;  // too slow for the ratio to mean anything
  const double ratio = v_ref / v_wheel;
  if (!(ratio > 0.5 && ratio < 2.0)) return;

  // Formulate it as a measurement instead of applying the ratio by hand.
  // z = v_ref, h(x) = b * v_wheel, so the innovation is y = v_ref - b*v_wheel
  // and the only non-zero Jacobian entry is dh/db = v_wheel.
  //
  // Applying `b *= ratio` on every call was a multiplicative feedback loop: the
  // reference arrives at 50 Hz, so b was multiplied by 0.9709 a hundred times
  // (0.9709^100 = 0.05) and slammed into the 0.80 clamp, which dragged v down to
  // 0.80 * 8.24 = 6.59 m/s and the path with it. Hand-applying a correction
  // also throws away the covariance, so nothing ever told the filter it had
  // already absorbed the information. Through the normal update the innovation
  // goes to zero as b converges, P(kB,kB) collapses, the Kalman gain vanishes,
  // and the remaining calls are no-ops.
  const double y = v_ref - x_[kB] * v_wheel;
  const double h_b = v_wheel;
  scalarUpdate(3, y, 0.0, 0.0, 0.0, h_b, 0.0, 0.0, op_.r_gnss_vel);
}


bool Observer::updateGnssVelocity(double v_meas) {
  const double y = v_meas - x_[kV];
  return scalarUpdate(0, y, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, op_.r_gnss_vel);
}

bool Observer::updateGnssPath(double s_meas, double s_meas_cov) {
  const double y = s_meas - x_[kS];
  const bool ok =
      scalarUpdate(2, y, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, std::max(1e-3, s_meas_cov));
  return ok;
}

void Observer::calibrateScale(double s_odometry, double s_reference) {
  if (std::fabs(s_odometry) < 1.0) return;  // not enough travel yet
  const double ratio = s_reference / s_odometry;
  if (!(ratio > 0.5 && ratio < 2.0)) return;
  x_[kB] = std::clamp(x_[kB] * ratio, 0.80, 1.25);
  P_(kB, kB) = op_.sigma_scale * op_.sigma_scale;
  // The accumulated distance is now expressed in the calibrated scale.
  x_[kS] = s_reference;
  P_(kS, kS) = std::max(1e-3, op_.r_gnss_pos);
}

void Observer::applyScaleBias(double log_scale, double dt) {
  if (!std::isfinite(log_scale) || dt <= 0.0) return;
  // 1/e time constant of ~0.5 s: fast enough to track a real systematic wheel
  // error, slow enough that a single noisy cycle cannot step the state.
  const double rate = std::clamp(dt / 0.5, 0.0, 1.0);
  if (rate <= 0.0) return;
  const double b_prev = x_[kB];
  x_[kB] = std::clamp(b_prev * std::exp(log_scale * rate), 0.80, 1.25);
  const double applied = std::fabs(std::log(x_[kB] / std::max(1e-6, b_prev)));
  if (applied > 1e-9) {
    // Cover the injected correction with the covariance, otherwise the filter
    // becomes overconfident about a state it did not measure.
    P_(kB, kB) += applied * applied;
    P_(kB, kB) = std::max(P_(kB, kB), 1e-8);
  }
}

}  // namespace tram
