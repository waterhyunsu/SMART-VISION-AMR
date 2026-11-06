#ifndef SMART_VISION_AMR_DWB__COMMAND_ENVELOPE_HPP_
#define SMART_VISION_AMR_DWB__COMMAND_ENVELOPE_HPP_

#include <algorithm>
#include <cmath>

namespace smart_vision_amr_dwb
{

// Bound translating turns before any trajectory critic evaluates them. Pure
// pivots retain the configured DWB yaw range; Nav2 Spin is a separate action.
// This rejects a curve instead of changing its curvature after scoring.
inline bool movingTurnAllowed(
  double linear_x, double linear_y, double yaw_rate, double moving_max_angular = 0.35,
  double measured_linear_speed = 0.0)
{
  if (!std::isfinite(linear_x) || !std::isfinite(linear_y) ||
    !std::isfinite(yaw_rate) || !std::isfinite(moving_max_angular) ||
    !std::isfinite(measured_linear_speed) ||
    moving_max_angular <= 0.0)
  {
    return false;
  }
  return std::max(std::hypot(linear_x, linear_y), std::abs(measured_linear_speed)) <= 0.01 + 1.e-9 ||
         std::abs(yaw_rate) <= moving_max_angular + 1.e-9;
}

// The same continuous limits are enforced by the physical motion mux. Reject
// candidates before collision scoring rather than changing their curvature
// after DWB has evaluated a different trajectory.
inline double turnLinearLimit(
  double yaw_rate, double straight_limit, double start_angular = 0.08,
  double full_angular = 0.35, double entry_linear = 0.15, double minimum_linear = 0.10)
{
  const double yaw = std::abs(yaw_rate);
  if (yaw < start_angular) {
    return straight_limit - (yaw / start_angular) * (straight_limit - entry_linear);
  }
  const double fraction = std::min(1.0, (yaw - start_angular) / (full_angular - start_angular));
  return entry_linear - fraction * (entry_linear - minimum_linear);
}

}  // namespace smart_vision_amr_dwb
#endif  // SMART_VISION_AMR_DWB__COMMAND_ENVELOPE_HPP_
