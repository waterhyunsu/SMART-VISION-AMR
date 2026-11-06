#include <cmath>
#include <initializer_list>
#include <limits>
#include <utility>

#include "gtest/gtest.h"
#include "smart_vision_amr_dwb/route_horizon.hpp"
#include "smart_vision_amr_dwb/command_envelope.hpp"
#include "smart_vision_amr_dwb/route_adherence.hpp"

namespace
{
using smart_vision_amr_dwb::limitRouteHorizon;
using smart_vision_amr_dwb::RouteHorizonStats;

nav_2d_msgs::msg::Path2D path(std::initializer_list<std::pair<double, double>> points)
{
  nav_2d_msgs::msg::Path2D result;
  result.header.frame_id = "odom";
  for (const auto & point : points) {
    geometry_msgs::msg::Pose2D pose;
    pose.x = point.first;
    pose.y = point.second;
    result.poses.push_back(pose);
  }
  return result;
}

geometry_msgs::msg::Pose2D robot(double x, double y)
{
  geometry_msgs::msg::Pose2D result;
  result.x = x;
  result.y = y;
  return result;
}

TEST(RouteHorizon, FoldedRouteDoesNotTargetNearbyReturnBranch)
{
  // All vertices fit inside a 0.75 m Euclidean radius, including the final
  // goal across a wall. The required route first travels down, then around.
  const auto route = path({{0., .3}, {0., 0.}, {0., -.6}, {-.5, -.6}, {-.5, 0.}});
  RouteHorizonStats stats;
  const auto limited = limitRouteHorizon(route, robot(0., 0.), .4, .85, &stats);
  ASSERT_EQ(limited.poses.size(), 3u);
  EXPECT_NEAR(limited.poses.back().x, 0., 1.e-10);
  EXPECT_NEAR(limited.poses.back().y, -.4, 1.e-10);
  EXPECT_EQ(limited.header.frame_id, "odom");
  EXPECT_TRUE(stats.cropped);
  EXPECT_NEAR(stats.anchor_arc_m, .3, 1.e-10);
  EXPECT_NEAR(stats.scoring_arc_m, .7, 1.e-10);
  // The controller's original route is retained; this function must not
  // replace the final action goal with a local endpoint.
  EXPECT_NEAR(route.poses.back().x, -.5, 1.e-10);
}

TEST(RouteHorizon, StraightRouteStartsHorizonAtProjectedProgress)
{
  const auto route = path({{0., 0.}, {.5, 0.}, {1.5, 0.}});
  const auto limited = limitRouteHorizon(route, robot(.4, .03), .55, .85);
  EXPECT_NEAR(limited.poses.back().x, .95, 1.e-10);
  EXPECT_NEAR(limited.poses.back().y, 0., 1.e-10);
}

TEST(RouteHorizon, CornerPreservesRouteSegmentsRatherThanCuttingDiagonal)
{
  const auto route = path({{0., 0.}, {.3, 0.}, {.3, .7}});
  const auto limited = limitRouteHorizon(route, robot(.1, 0.), .4, .85);
  ASSERT_EQ(limited.poses.size(), 3u);
  EXPECT_NEAR(limited.poses[1].x, .3, 1.e-10);
  EXPECT_NEAR(limited.poses[1].y, 0., 1.e-10);
  EXPECT_NEAR(limited.poses.back().x, .3, 1.e-10);
  EXPECT_NEAR(limited.poses.back().y, .2, 1.e-10);
}

TEST(RouteHorizon, NearbyReturnAtExactCrossingCannotAdvanceProgress)
{
  const auto route = path({{0., 0.}, {.5, 0.}, {.5, .5}, {0., .5}, {0., 0.}, {-.5, 0.}});
  const auto limited = limitRouteHorizon(route, robot(0., 0.), .4, .85);
  EXPECT_NEAR(limited.poses.back().x, .4, 1.e-10);
  EXPECT_NEAR(limited.poses.back().y, 0., 1.e-10);
}

TEST(RouteHorizon, AnchorSearchCannotJumpOntoLaterBranchSlightlyCloser)
{
  const auto route = path({{0., 0.}, {.6, 0.}, {.6, .3}, {.1, .3}, {.1, .02}});
  const auto limited = limitRouteHorizon(route, robot(.1, .02), .4, .65);
  EXPECT_NEAR(limited.poses.back().x, .5, 1.e-10);
  EXPECT_NEAR(limited.poses.back().y, 0., 1.e-10);
}

TEST(RouteHorizon, ArrivalRetainsFinalPoseAndYaw)
{
  auto route = path({{0., 0.}, {.15, 0.}, {.2, 0.}});
  route.poses.back().theta = 1.5;
  RouteHorizonStats stats;
  const auto limited = limitRouteHorizon(route, robot(.16, .01), .55, .85, &stats);
  ASSERT_EQ(limited.poses.size(), route.poses.size());
  EXPECT_NEAR(limited.poses.back().x, .2, 1.e-10);
  EXPECT_NEAR(limited.poses.back().theta, 1.5, 1.e-10);
  EXPECT_FALSE(stats.cropped);
}

TEST(RouteHorizon, DuplicateAndEmptyPosesDoNotProduceNonfiniteCoordinates)
{
  auto empty = path({});
  EXPECT_TRUE(limitRouteHorizon(empty, robot(0., 0.), .55, .85).poses.empty());
  const auto duplicate = path({{0., 0.}, {0., 0.}, {.9, 0.}});
  const auto limited = limitRouteHorizon(duplicate, robot(.1, 0.), .55, .85);
  EXPECT_NEAR(limited.poses.back().x, .65, 1.e-10);
  EXPECT_TRUE(std::isfinite(limited.poses.back().y));
}

TEST(RouteHorizon, RotatedResampledRoutePreservesMetricHorizon)
{
  auto coarse = path({{0., 0.}, {0., 1.}});
  auto fine = path({{0., 0.}, {0., .2}, {0., .4}, {0., .6}, {0., .8}, {0., 1.}});
  const auto a = limitRouteHorizon(coarse, robot(.03, .3), .55, .85);
  const auto b = limitRouteHorizon(fine, robot(.03, .3), .55, .85);
  EXPECT_NEAR(a.poses.back().y, .85, 1.e-10);
  EXPECT_NEAR(a.poses.back().y, b.poses.back().y, 1.e-10);
}

TEST(CommandEnvelope, TurnSpeedMatchesMuxWithoutChangingYaw)
{
  using smart_vision_amr_dwb::turnLinearLimit;
  EXPECT_DOUBLE_EQ(turnLinearLimit(0., .22), .22);
  EXPECT_NEAR(turnLinearLimit(.04, .22), .185, 1.e-12);
  EXPECT_NEAR(turnLinearLimit(.08, .22), .15, 1.e-12);
  EXPECT_NEAR(turnLinearLimit(.215, .22), .125, 1.e-12);
  EXPECT_NEAR(turnLinearLimit(.35, .22), .10, 1.e-12);
  EXPECT_NEAR(turnLinearLimit(-.5, .22), .10, 1.e-12);
}

TEST(CommandEnvelope, RecordedTranslatingTurnIsRejectedBeforeScoring)
{
  using smart_vision_amr_dwb::movingTurnAllowed;
  // Actual cold/source-rate trial: this curve overshot the unchanged stop QA.
  EXPECT_FALSE(movingTurnAllowed(.09263, 0., .47436));
  EXPECT_FALSE(movingTurnAllowed(.09263, 0., -.47436));
  EXPECT_TRUE(movingTurnAllowed(.10, 0., .35));
  EXPECT_TRUE(movingTurnAllowed(.10, 0., -.35));
  EXPECT_TRUE(movingTurnAllowed(.22, 0., 0.));
  EXPECT_TRUE(movingTurnAllowed(0., 0., .5));
  EXPECT_TRUE(movingTurnAllowed(0., 0., -.5));
}

TEST(CommandEnvelope, TranslationAndYawBoundariesAndCustomCap)
{
  using smart_vision_amr_dwb::movingTurnAllowed;
  EXPECT_TRUE(movingTurnAllowed(.01, 0., .5));
  EXPECT_FALSE(movingTurnAllowed(.010001, 0., .5));
  EXPECT_FALSE(movingTurnAllowed(-.010001, 0., -.5));
  EXPECT_FALSE(movingTurnAllowed(0., .010001, .5));
  EXPECT_FALSE(movingTurnAllowed(.008, .008, .5));
  EXPECT_FALSE(movingTurnAllowed(.10, 0., .350001));
  EXPECT_TRUE(movingTurnAllowed(.10, 0., .4, .4));
  EXPECT_FALSE(movingTurnAllowed(.10, 0., .400001, .4));
}

TEST(CommandEnvelope, PurePivotCandidateCannotIgnoreMeasuredLinearCoast)
{
  using smart_vision_amr_dwb::movingTurnAllowed;
  EXPECT_FALSE(movingTurnAllowed(0., 0., .5, .35, .22));
  EXPECT_FALSE(movingTurnAllowed(0., 0., -.5, .35, -.10));
  EXPECT_TRUE(movingTurnAllowed(0., 0., .35, .35, .22));
  EXPECT_TRUE(movingTurnAllowed(0., 0., .5, .35, .01));
  EXPECT_FALSE(movingTurnAllowed(0., 0., .5, .35,
    std::numeric_limits<double>::quiet_NaN()));
}

TEST(CommandEnvelope, InvalidCommandsOrLimitNeverReachCritics)
{
  using smart_vision_amr_dwb::movingTurnAllowed;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(movingTurnAllowed(nan, 0., 0.));
  EXPECT_FALSE(movingTurnAllowed(0., nan, 0.));
  EXPECT_FALSE(movingTurnAllowed(0., 0., nan));
  EXPECT_FALSE(movingTurnAllowed(.1, 0., inf));
  EXPECT_FALSE(movingTurnAllowed(.1, 0., .1, 0.));
  EXPECT_FALSE(movingTurnAllowed(.1, 0., .1, -1.));
  EXPECT_FALSE(movingTurnAllowed(.1, 0., .1, nan));
  EXPECT_FALSE(movingTurnAllowed(.1, 0., .1, inf));
}

std::vector<geometry_msgs::msg::Pose2D> curve(
  double x, double y, double yaw, double v, double w)
{
  std::vector<geometry_msgs::msg::Pose2D> points;
  for (int i = 0; i <= 40; ++i) {
    const double t = .05 * i;
    auto p = robot(x, y);
    p.theta = yaw + w * t;
    if (std::abs(w) > 1.e-9) {
      p.x += v / w * (std::sin(yaw + w * t) - std::sin(yaw));
      p.y -= v / w * (std::cos(yaw + w * t) - std::cos(yaw));
    } else {
      p.x += v * t * std::cos(yaw);
      p.y += v * t * std::sin(yaw);
    }
    points.push_back(p);
  }
  return points;
}

TEST(RouteAdherence, RecordedPermutedApproachPrefersCorrectionBeforeTheDistantBend)
{
  // Actual published path t82.5917, stamp62.12s; map pose reconstructed from
  // sampled odom + map->odom at t82.98 (not an original tf2-buffer snapshot).
  const auto route = path({{.075, -1.025}, {.575, -1.125}, {.59, -1.1625},
      {.65, -1.3125}, {.675, -1.6562}});
  const auto inward = curve(.072, -1.010, -.557, .16, -.15);
  const auto correction = curve(.072, -1.010, -.557, .16, .20);
  const auto no_correction = curve(.072, -1.010, -.557, .16, 0.);
  const auto score = [&](const auto & candidate) {
      return smart_vision_amr_dwb::routeAdherenceCost(route, candidate, 0.);
    };
  EXPECT_GT(score(inward), score(no_correction));
  EXPECT_GT(score(no_correction), score(correction));
  EXPECT_GT(32. * (score(inward) - score(correction)), .5);
  EXPECT_EQ(route.poses.back().x, .675);  // No path or goal replacement.
}

TEST(RouteAdherence, OrderedFoldedRouteCannotSkipToNearbyReturnBranch)
{
  const auto route = path({{0., 0.}, {.8, 0.}, {.8, .1}, {0., .1}});
  const auto correct = curve(0., 0., 0., .15, 0.);
  const auto shortcut = curve(0., .1, std::acos(-1.), .15, 0.);
  EXPECT_LT(smart_vision_amr_dwb::routeAdherenceCost(route, correct, 0.),
    smart_vision_amr_dwb::routeAdherenceCost(route, shortcut, 0.));
}

TEST(RouteAdherence, ResamplingAndPrunedPrefixHaveTheSameMetricScore)
{
  const auto sparse = path({{0., 0.}, {1., 0.}});
  const auto dense = path({{0., 0.}, {.1, 0.}, {.2, 0.}, {.3, 0.}, {1., 0.}});
  const auto pruned = path({{.2, 0.}, {1., 0.}});
  const auto candidate = curve(.2, .02, .1, .15, -.1);
  const auto score = smart_vision_amr_dwb::routeAdherenceCost(sparse, candidate, .2);
  EXPECT_NEAR(score, smart_vision_amr_dwb::routeAdherenceCost(dense, candidate, .2), 1.e-12);
  EXPECT_NEAR(score, smart_vision_amr_dwb::routeAdherenceCost(pruned, candidate, 0.), 1.e-12);
}

TEST(RouteAdherence, EmptyAndDegeneratePathsDoNotInventDirectionAndInvalidPoseIsRejected)
{
  const auto candidate = curve(0., 0., 0., .15, 0.);
  EXPECT_EQ(0., smart_vision_amr_dwb::routeAdherenceCost(path({}), candidate, 0.));
  EXPECT_EQ(0., smart_vision_amr_dwb::routeAdherenceCost(path({{0., 0.}, {0., 0.}}), candidate, 0.));
  auto invalid = candidate;
  invalid.front().theta = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(std::isfinite(smart_vision_amr_dwb::routeAdherenceCost(
    path({{0., 0.}, {1., 0.}}), invalid, 0.)));
}
}  // namespace
