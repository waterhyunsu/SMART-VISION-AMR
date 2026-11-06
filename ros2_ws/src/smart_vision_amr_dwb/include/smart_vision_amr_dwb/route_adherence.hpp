#ifndef SMART_VISION_AMR_DWB__ROUTE_ADHERENCE_HPP_
#define SMART_VISION_AMR_DWB__ROUTE_ADHERENCE_HPP_

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include "geometry_msgs/msg/pose2_d.hpp"
#include "nav_2d_msgs/msg/path2_d.hpp"

namespace smart_vision_amr_dwb
{

// A near-field metric cost complements the costmap critics' distant carrot.
// It follows ordered route progress; a folded return branch is not a shortcut.
// No candidate is made collision-free by this cost and the action goal is intact.
inline double routeAdherenceCost(
  const nav_2d_msgs::msg::Path2D & route,
  const std::vector<geometry_msgs::msg::Pose2D> & trajectory,
  double anchor_arc, double prefix_distance = 0.30, double heading_distance = 0.15)
{
  if (route.poses.size() < 2 || trajectory.empty()) {
    return 0.0;
  }
  double traveled = 0.0;
  double progress = std::max(0.0, anchor_arc - 0.03);
  double total = 0.0;
  size_t count = 0;
  for (size_t i = 0; i < trajectory.size(); ++i) {
    const auto & pose = trajectory[i];
    if (!std::isfinite(pose.x) || !std::isfinite(pose.y) || !std::isfinite(pose.theta)) {
      return std::numeric_limits<double>::infinity();
    }
    if (i > 0) {
      traveled += std::hypot(pose.x - trajectory[i - 1].x, pose.y - trajectory[i - 1].y);
    }
    if (traveled > prefix_distance + 1.e-9) {
      break;
    }
    double arc = 0.0;
    double best = std::numeric_limits<double>::infinity();
    double tangent = 0.0;
    double best_progress = progress;
    // A short projection slack tolerates sampling; it cannot jump a corner or
    // folded branch farther along the plan than physically traveled distance.
    const double maximum = anchor_arc + traveled + 0.10;
    for (size_t j = 1; j < route.poses.size(); ++j) {
      const auto & a = route.poses[j - 1];
      const auto & b = route.poses[j];
      const double dx = b.x - a.x, dy = b.y - a.y;
      const double length = std::hypot(dx, dy);
      if (length <= 1.e-10) {
        continue;
      }
      if (arc > maximum) {
        break;
      }
      if (arc + length >= progress) {
        const double low = std::clamp((progress - arc) / length, 0.0, 1.0);
        const double high = std::clamp((maximum - arc) / length, low, 1.0);
        const double fraction = std::clamp(
          ((pose.x - a.x) * dx + (pose.y - a.y) * dy) / (length * length), low, high);
        const double distance = std::hypot(pose.x - a.x - fraction * dx,
            pose.y - a.y - fraction * dy);
        if (distance < best - 1.e-12) {
          best = distance;
          tangent = std::atan2(dy, dx);
          best_progress = arc + fraction * length;
        }
      }
      arc += length;
    }
    if (!std::isfinite(best)) {
      break;
    }
    progress = best_progress;
    const double error = std::abs(std::atan2(
        std::sin(pose.theta - tangent), std::cos(pose.theta - tangent)));
    total += best + heading_distance * error;
    ++count;
  }
  return count ? total / count : 0.0;
}

}  // namespace smart_vision_amr_dwb
#endif  // SMART_VISION_AMR_DWB__ROUTE_ADHERENCE_HPP_
