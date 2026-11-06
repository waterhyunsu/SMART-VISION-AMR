#include <cmath>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <utility>

#include "smart_vision_amr_dwb/route_tracking_dwb.hpp"
#include "smart_vision_amr_dwb/route_horizon.hpp"
#include "smart_vision_amr_dwb/command_envelope.hpp"
#include "smart_vision_amr_dwb/route_adherence.hpp"
#include "dwb_core/exceptions.hpp"
#include "nav2_util/node_utils.hpp"
#include "nav_2d_utils/tf_help.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace smart_vision_amr_dwb
{

void RouteTrackingDWB::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  auto node = parent.lock();
  if (!node) {
    throw std::runtime_error("RouteTrackingDWB requires an active lifecycle node");
  }
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".route_horizon_distance", rclcpp::ParameterValue(0.55));
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".route_anchor_search_distance", rclcpp::ParameterValue(0.85));
  node->get_parameter(name + ".route_horizon_distance", route_horizon_distance_);
  node->get_parameter(name + ".route_anchor_search_distance", route_anchor_search_distance_);
  for (const auto & parameter : {
      std::make_pair("turn_start_angular", &turn_start_angular_),
      std::make_pair("turn_full_angular", &turn_full_angular_),
      std::make_pair("turn_entry_linear", &turn_entry_linear_),
      std::make_pair("turn_min_linear", &turn_min_linear_),
      std::make_pair("route_adherence_scale", &route_adherence_scale_),
      std::make_pair("moving_turn_max_angular", &moving_turn_max_angular_)})
  {
    nav2_util::declare_parameter_if_not_declared(
      node, name + "." + parameter.first, rclcpp::ParameterValue(*parameter.second));
    node->get_parameter(name + "." + parameter.first, *parameter.second);
  }
  if (!std::isfinite(route_horizon_distance_) || route_horizon_distance_ <= 0.0 ||
    !std::isfinite(route_anchor_search_distance_) || route_anchor_search_distance_ <= 0.0)
  {
    throw std::invalid_argument("Route horizon and anchor search distances must be finite and positive");
  }
  dwb_core::DWBLocalPlanner::configure(parent, name, std::move(tf), std::move(costmap_ros));
  node->get_parameter(name + ".max_speed_xy", nominal_linear_limit_);
  node->get_parameter(name + ".xy_goal_tolerance", route_goal_tolerance_);
  if (!std::isfinite(nominal_linear_limit_) || nominal_linear_limit_ <= 0.0 ||
    !std::isfinite(turn_start_angular_) || !std::isfinite(turn_full_angular_) ||
    !std::isfinite(turn_entry_linear_) || !std::isfinite(turn_min_linear_) ||
    turn_start_angular_ <= 0.0 || turn_full_angular_ <= turn_start_angular_ ||
    turn_min_linear_ <= 0.0 || turn_entry_linear_ < turn_min_linear_ ||
    nominal_linear_limit_ < turn_entry_linear_ ||
    !std::isfinite(moving_turn_max_angular_) || moving_turn_max_angular_ <= 0.0 ||
    !std::isfinite(route_adherence_scale_) || route_adherence_scale_ < 0.0)
  {
    throw std::invalid_argument("DWB command envelope has invalid speed bounds");
  }
  diagnostics_pub_ = node->create_publisher<std_msgs::msg::String>(
    "/smart_vision_amr/controller_diagnostics", 1);
  RCLCPP_INFO(
    node->get_logger(), "DWB local route horizon: %.2f m along path; anchor search: %.2f m",
    route_horizon_distance_, route_anchor_search_distance_);
}

nav_2d_msgs::msg::Twist2DStamped RouteTrackingDWB::computeVelocityCommands(
  const nav_2d_msgs::msg::Pose2DStamped & pose,
  const nav_2d_msgs::msg::Twist2D & velocity,
  std::shared_ptr<dwb_msgs::msg::LocalPlanEvaluation> & results)
{
  measured_linear_speed_ = std::hypot(velocity.x, velocity.y);
  return dwb_core::DWBLocalPlanner::computeVelocityCommands(pose, velocity, results);
}

dwb_msgs::msg::TrajectoryScore RouteTrackingDWB::scoreTrajectory(
  const dwb_msgs::msg::Trajectory2D & trajectory, double best_score)
{
  const auto & command = trajectory.velocity;
  if (!movingTurnAllowed(command.x, command.y, command.theta, moving_turn_max_angular_,
      measured_linear_speed_)) {
    throw dwb_core::IllegalTrajectoryException(
            "CommandEnvelope", "Exceeds translating turn yaw limit or has nonfinite command");
  }
  if (std::hypot(command.x, command.y) > linear_speed_limit_.load() + 1.e-9) {
    throw dwb_core::IllegalTrajectoryException("CommandEnvelope", "Exceeds active linear speed limit");
  }
  if (command.x > 0.0 && command.x > turnLinearLimit(
      command.theta, nominal_linear_limit_, turn_start_angular_, turn_full_angular_,
      turn_entry_linear_, turn_min_linear_) + 1.e-9)
  {
    throw dwb_core::IllegalTrajectoryException("CommandEnvelope", "Exceeds autonomous turn speed limit");
  }
  auto score = dwb_core::DWBLocalPlanner::scoreTrajectory(trajectory, best_score);
  if (route_adherence_active_ && route_adherence_scale_ > 0.0) {
    dwb_msgs::msg::CriticScore adherence;
    adherence.name = "RouteAdherence";
    adherence.scale = route_adherence_scale_;
    adherence.raw_score = routeAdherenceCost(
      route_adherence_reference_, trajectory.poses, route_anchor_arc_);
    if (!std::isfinite(adherence.raw_score)) {
      throw dwb_core::IllegalTrajectoryException("RouteAdherence", "Nonfinite trajectory pose");
    }
    score.total += adherence.raw_score * adherence.scale;
    score.scores.push_back(adherence);
  }
  return score;
}

void RouteTrackingDWB::setSpeedLimit(const double & speed_limit, const bool & percentage)
{
  // Nav2's NO_SPEED_LIMIT is zero. Keep yaw sampling unchanged: a linear
  // curve-speed request must not reduce steering authority on a tight turn.
  if (!std::isfinite(speed_limit) || speed_limit < 0.0) {
    return;
  }
  if (speed_limit == 0.0) {
    linear_speed_limit_.store(std::numeric_limits<double>::infinity());
  } else {
    linear_speed_limit_.store(
      percentage ? nominal_linear_limit_ * std::min(100.0, speed_limit) / 100.0 : speed_limit);
  }
}

void RouteTrackingDWB::activate()
{
  dwb_core::DWBLocalPlanner::activate();
  diagnostics_pub_->on_activate();
  last_diagnostic_ns_ = 0;
}

void RouteTrackingDWB::deactivate()
{
  diagnostics_pub_->on_deactivate();
  dwb_core::DWBLocalPlanner::deactivate();
}

void RouteTrackingDWB::cleanup()
{
  diagnostics_pub_.reset();
  dwb_core::DWBLocalPlanner::cleanup();
}

nav_2d_msgs::msg::Path2D RouteTrackingDWB::transformGlobalPlan(
  const nav_2d_msgs::msg::Pose2DStamped & pose)
{
  const auto original_count = global_plan_.poses.size();
  auto transformed = dwb_core::DWBLocalPlanner::transformGlobalPlan(pose);
  nav_2d_msgs::msg::Pose2DStamped local_pose;
  if (!nav_2d_utils::transformPose(
      tf_, transformed.header.frame_id, pose, local_pose, transform_tolerance_))
  {
    throw dwb_core::PlannerTFException("Unable to transform robot pose for route horizon");
  }
  RouteHorizonStats stats;
  auto horizon = limitRouteHorizon(
    transformed, local_pose.pose, route_horizon_distance_, route_anchor_search_distance_, &stats);
  route_adherence_reference_ = horizon;
  route_anchor_arc_ = stats.anchor_arc_m;
  // The real goal's yaw belongs to RotateToGoal. A route-tangent cost must not
  // prevent arrival or an in-place final heading adjustment.
  route_adherence_active_ = horizon.poses.size() >= 2 && (stats.cropped ||
    std::hypot(local_pose.pose.x - horizon.poses.back().x,
    local_pose.pose.y - horizon.poses.back().y) > route_goal_tolerance_);
  const auto now_ns = clock_->now().nanoseconds();
  if (diagnostics_pub_ && diagnostics_pub_->is_activated() &&
    (last_diagnostic_ns_ == 0 || now_ns < last_diagnostic_ns_ ||
    now_ns - last_diagnostic_ns_ >= 1000000000LL))
  {
    last_diagnostic_ns_ = now_ns;
    std::ostringstream json;
    json << std::fixed << std::setprecision(5)
         << "{\"schema_version\":1,\"sent_at_ns\":" << now_ns
         << ",\"original_pose_count\":" << original_count
         << ",\"transformed_pose_count\":" << transformed.poses.size()
         << ",\"horizon_pose_count\":" << horizon.poses.size()
         << ",\"anchor_arc_m\":" << stats.anchor_arc_m
         << ",\"scoring_arc_m\":" << stats.scoring_arc_m
         << ",\"horizon_m\":" << route_horizon_distance_
         << ",\"moving_turn_max_angular_radps\":" << moving_turn_max_angular_
         << ",\"measured_linear_speed_mps\":" << measured_linear_speed_
         << ",\"route_adherence_scale\":" << route_adherence_scale_
         << ",\"route_adherence_active\":" << (route_adherence_active_ ? "true" : "false")
         << ",\"linear_speed_limit_mps\":"
         << std::min(nominal_linear_limit_, linear_speed_limit_.load())
         << ",\"cropped\":" << (stats.cropped ? "true" : "false");
    if (!horizon.poses.empty()) {
      json << ",\"local_goal_x\":" << horizon.poses.back().x
           << ",\"local_goal_y\":" << horizon.poses.back().y;
    }
    json << "}";
    std_msgs::msg::String message;
    message.data = json.str();
    diagnostics_pub_->publish(message);
  }
  return horizon;
}

}  // namespace smart_vision_amr_dwb

PLUGINLIB_EXPORT_CLASS(smart_vision_amr_dwb::RouteTrackingDWB, nav2_core::Controller)
