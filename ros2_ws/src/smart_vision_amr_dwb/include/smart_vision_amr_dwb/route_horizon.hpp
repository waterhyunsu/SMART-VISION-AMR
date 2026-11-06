#ifndef SMART_VISION_AMR_DWB__ROUTE_HORIZON_HPP_
#define SMART_VISION_AMR_DWB__ROUTE_HORIZON_HPP_

#include <algorithm>
#include <cmath>
#include <limits>

#include "geometry_msgs/msg/pose2_d.hpp"
#include "nav_2d_msgs/msg/path2_d.hpp"

namespace smart_vision_amr_dwb
{

struct RouteHorizonStats
{
  double anchor_arc_m{0.0};
  double scoring_arc_m{0.0};
  bool cropped{false};
};

// DWB's forward_prune_distance is a Euclidean radius. In a U-shaped route it
// can include the return branch and actual goal across a wall. Bound only the
// critic's transformed path by arc length; never change the action's goal.
inline nav_2d_msgs::msg::Path2D limitRouteHorizon(
  const nav_2d_msgs::msg::Path2D & path, const geometry_msgs::msg::Pose2D & robot,
  double forward_distance, double anchor_search_distance, RouteHorizonStats * stats = nullptr)
{
  if (stats) {
    *stats = RouteHorizonStats{};
  }
  if (path.poses.size() < 2) {
    return path;
  }
  double anchor_arc = 0.0;
  double best_distance_sq = std::numeric_limits<double>::infinity();
  double arc = 0.0;
  for (size_t index = 1; index < path.poses.size(); ++index) {
    const auto & start = path.poses[index - 1];
    const auto & end = path.poses[index];
    const double dx = end.x - start.x;
    const double dy = end.y - start.y;
    const double length_sq = dx * dx + dy * dy;
    const double length = std::sqrt(length_sq);
    if (length < 1.e-10) {
      continue;
    }
    if (arc > anchor_search_distance) {
      break;
    }
    const double max_fraction = std::min(1.0, (anchor_search_distance - arc) / length);
    const double fraction = std::clamp(
      ((robot.x - start.x) * dx + (robot.y - start.y) * dy) / length_sq,
      0.0, max_fraction);
    const double error_x = robot.x - start.x - fraction * dx;
    const double error_y = robot.y - start.y - fraction * dy;
    const double distance_sq = error_x * error_x + error_y * error_y;
    // Preserve the earliest branch when folded/crossing segments tie.
    if (distance_sq < best_distance_sq - 1.e-12) {
      best_distance_sq = distance_sq;
      anchor_arc = arc + fraction * length;
    }
    arc += length;
  }

  const double end_arc = anchor_arc + forward_distance;
  if (stats) {
    stats->anchor_arc_m = anchor_arc;
  }
  nav_2d_msgs::msg::Path2D result;
  result.header = path.header;
  result.poses.push_back(path.poses.front());
  arc = 0.0;
  for (size_t index = 1; index < path.poses.size(); ++index) {
    const auto & start = path.poses[index - 1];
    const auto & end = path.poses[index];
    const double length = std::hypot(end.x - start.x, end.y - start.y);
    if (arc + length > end_arc && length > 1.e-10) {
      const double fraction = std::clamp((end_arc - arc) / length, 0.0, 1.0);
      auto endpoint = start;
      endpoint.x += fraction * (end.x - start.x);
      endpoint.y += fraction * (end.y - start.y);
      const double yaw_delta = std::atan2(
        std::sin(end.theta - start.theta), std::cos(end.theta - start.theta));
      endpoint.theta += fraction * yaw_delta;
      result.poses.push_back(endpoint);
      if (stats) {
        stats->cropped = true;
        stats->scoring_arc_m = end_arc;
      }
      return result;
    }
    result.poses.push_back(end);
    arc += length;
  }
  if (stats) {
    stats->scoring_arc_m = arc;
  }
  return result;
}

}  // namespace smart_vision_amr_dwb
#endif  // SMART_VISION_AMR_DWB__ROUTE_HORIZON_HPP_
