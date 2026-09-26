// Shared plain-old-data helpers that need a translation unit.
#include "tram_odometry/types.hpp"

namespace tram {

const char* to_string(SlipReason r) {
  switch (r) {
    case SlipReason::kNone:
      return "none";
    case SlipReason::kBogieMismatch:
      return "bogie_mismatch";
    case SlipReason::kYawMismatch:
      return "yaw_mismatch";
    case SlipReason::kTorqueAccelConflict:
      return "torque_accel_conflict";
    case SlipReason::kFrozenSensor:
      return "frozen_sensor";
    case SlipReason::kDropout:
      return "dropout";
    case SlipReason::kAccelBeyondAdhesion:
      return "accel_beyond_adhesion";
    case SlipReason::kOutlier:
      return "outlier";
  }
  return "unknown";
}

}  // namespace tram
