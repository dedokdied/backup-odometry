#include "tram_odometry/slip_detector.hpp"

#include <algorithm>
#include <cmath>

namespace tram {
namespace {

double ramp(double x, double lo, double hi) {
  if (hi <= lo) return x > hi ? 1.0 : 0.0;
  return std::clamp((x - lo) / (hi - lo), 0.0, 1.0);
}

}  // namespace

SlipDetector::SlipDetector(const AdhesionParams& ap, const FilterParams& fp,
                           const VehicleParams& vp)
    : ap_(ap), fp_(fp), vp_(vp), trust_lp_(2.0, 0.02), slip_lp_(5.0, 0.02) {
  reset();
}

void SlipDetector::reset() {
  trust_lp_.reset();
  slip_lp_.reset();
  trust_ = 0.5;
  slip_index_ = 0.0;
  slip_ratio_ = 0.0;
  mu_ = ap_.mu_peak;
  reason_ = SlipReason::kNone;
}

double SlipDetector::update(const SlipFeatures& f, double dt) {
  if (dt <= 0.0) dt = 0.02;

  double penalty = 0.0;
  double slip_evidence = 0.0;
  SlipReason reason = SlipReason::kNone;
  auto note = [&](double p, SlipReason r, double slip_p = 0.0) {
    if (p > penalty) {
      penalty = p;
      reason = r;
    }
    slip_evidence = std::max(slip_evidence, slip_p);
  };

  // --- 1. frozen / missing sensors -----------------------------------------
  if (f.dropout || !f.front_valid || !f.rear_valid) {
    note(1.0, f.dropout ? SlipReason::kDropout : SlipReason::kNone);
  }

  // A constant wheel value is only evidence of a stuck sensor when the wheels
  // are actually changing. The discriminator has to be the *measured* wheel
  // acceleration: the model acceleration is never exactly zero, because rolling
  // and aerodynamic resistance keep it near -0.07 m/s^2 even in a steady cruise,
  // so gating on the model condemns every constant-speed run. A tram holding a
  // speed emits a bit-exactly constant tachometer, and penalising that zeroed
  // the trust for the whole bag.
  const bool motion_expected =
      std::fabs(f.a_wheel) > 0.05 || f.drive_force > 1.0 || f.brake_force > 1.0;
  if (motion_expected) {
    if (f.frozen_front) note(1.0, SlipReason::kFrozenSensor);
    if (f.frozen_rear) note(1.0, SlipReason::kFrozenSensor);
  }

  // --- 2. front/rear mismatch ------------------------------------------------
  // All wheels are driven, so in nominal condition the two bogies agree.
  if (f.front_valid && f.rear_valid) {
    const double mismatch = std::fabs(f.v_front - f.v_rear);
    const double kinematic = 2.0 * f.wheelbase * std::fabs(f.yaw_rate);
    const double resid_nominal = ramp(mismatch, ap_.bogie_mismatch_warn, ap_.bogie_mismatch_fault);
    note(resid_nominal, SlipReason::kBogieMismatch);

    // When yaw is known, part of the difference is explained by turning.
    if (std::fabs(f.yaw_rate) > 1e-3) {
      const double resid = std::fabs(mismatch - kinematic);
      const double p = ramp(resid, ap_.bogie_mismatch_warn, ap_.bogie_mismatch_fault);
      note(p, SlipReason::kYawMismatch, p);
    }
  }

  // --- 0. kinematic envelope (mu-independent) -------------------------------
  // A 38 t tram cannot accelerate or brake harder than a couple of m/s^2 no
  // matter what the adhesion model claims, so this bounds the sensor chain
  // itself instead of relying on a tuned mu.
  if (std::fabs(f.a_wheel) > 2.5) {
    note(1.0, SlipReason::kAccelBeyondAdhesion, 1.0);
  }

  // --- 3. measured acceleration beyond what adhesion can transmit -----------
  // This is the decisive test: it needs no extra sensors, only the model. The
  // ceiling must be computed against the same effective mass the model uses;
  // dividing by (mass_kg + rot_inertia_kgm2) added kilograms to kg*m^2 and gave
  // a slightly wrong threshold.
  if (f.front_valid && f.rear_valid && f.adhesion_limit > 1.0) {
    const double a_ceiling = f.adhesion_limit / std::max(1.0, f.effective_mass);
    const double excess = std::fabs(f.a_wheel) - a_ceiling;
    if (excess > 0.15) {
      const double p = ramp(excess, 0.15, 0.8);
      note(p, SlipReason::kAccelBeyondAdhesion, p);
    }
  }

  // --- 4. high torque with little acceleration (wheelspin) ------------------
  // Thresholds are deliberately loose. During any brisk acceleration the model
  // trails the wheels by roughly the actuator lag (tau = 0.2 s), so a_model and
  // a_wheel are *expected* to disagree by a few tenths. With the old 0.25 m/s^2
  // threshold that lag was read as wheelspin on every launch, trust collapsed,
  // R was inflated and the filter stopped believing the wheels exactly when it
  // needed them. Real wheelspin is a much larger disagreement.
  if (f.driver_valid && f.adhesion_limit > 1.0) {
    const double demand = f.drive_force;
    const double utilisation = demand / f.adhesion_limit;
    if (utilisation > 0.85) {
      const double gain_error = f.a_model - f.a_wheel;  // model says more than measured
      const double kErrGate = 0.6;
      if (std::fabs(gain_error) > kErrGate && f.v_front > 0.5) {
        double p = ramp(std::fabs(gain_error), kErrGate, 1.5) * ramp(utilisation, 0.85, 1.0);
        // A hard transient is not evidence of slip: ease off while the speed is
        // changing quickly so start/stop passes do not destroy the trust.
        if (std::fabs(f.a_wheel) > 1.5) p *= 0.5;
        note(p, SlipReason::kTorqueAccelConflict, p);
      }
    }
  }

  // --- 5. braking with wheel lock (odometry drops below the model) ----------
  // Same reasoning as test 4: brake demand builds through its own lag, and the
  // model reacts later than the wheels. The gate was 0.3 m/s^2, which that lag
  // clears on every ordinary stop.
  if (f.brake_force > 100.0 && f.front_valid && f.rear_valid && f.v_front > 0.5) {
    const double lock = ramp(f.brake_force / std::max(1.0, f.adhesion_limit) - 1.0, 0.0, 0.6);
    const double kDevGate = 0.6;
    const double dev = ramp(f.a_model - f.a_wheel, kDevGate, 1.8);
    double p = lock * dev;
    if (std::fabs(f.a_wheel) > 1.5) p *= 0.5;
    note(p, SlipReason::kTorqueAccelConflict, p);
  }

  // Smooth the confidence so the filter covariance does not chatter.
  const double raw_trust = 1.0 - penalty;
  const double raw_slip = std::max(slip_evidence, penalty);
  trust_ = trust_lp_.process(raw_trust, dt);
  slip_index_ = slip_lp_.process(raw_slip, dt);
  trust_ = std::clamp(trust_, 0.0, 1.0);
  slip_index_ = std::clamp(slip_index_, 0.0, 1.0);
  reason_ = reason;

  // Slip ratio estimate: how much of the demanded force adhesion could not take.
  if (f.adhesion_limit > 1.0 && f.drive_force > f.adhesion_limit) {
    const double excess = f.drive_force / f.adhesion_limit - 1.0;
    slip_ratio_ = std::min(ap_.slip_hard, ap_.slip_peak + excess * 0.3);
  } else if (f.brake_force > f.adhesion_limit && f.v_front > 0.5) {
    const double excess = f.brake_force / f.adhesion_limit - 1.0;
    slip_ratio_ = std::min(ap_.slip_hard, ap_.slip_peak + excess * 0.3);
  } else {
    // Decay the estimate when no slip is evident.
    const double decay = std::exp(-dt / 0.5);
    slip_ratio_ *= decay;
  }
  slip_ratio_ = std::clamp(slip_ratio_, 0.0, ap_.slip_hard);

  return trust_;
}

}  // namespace tram
