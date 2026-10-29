"""Deployment settings that must agree across planner, controller and memory."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "ros2_ws/src/smart_vision_amr"


def test_classroom_memory_and_live_obstacle_rules_are_enabled():
    nodes = yaml.safe_load((PACKAGE / "config/classroom_hardware_nodes.yaml").read_text())
    planner = nodes["/astar_planner"]["ros__parameters"]
    memory = nodes["/change_memory"]["ros__parameters"]
    launcher = (ROOT / "scripts/run_classroom_navigation_16.sh").read_text()

    assert planner["live_obstacle_layer_required"] is True
    assert "mapped_wall_return_tolerance_cells" not in planner
    assert memory["mapped_wall_return_tolerance_cells"] == 1
    assert memory["confirm_observations"] == 5
    assert memory["confirm_min_duration_sec"] == 2.0
    assert memory["clear_observations"] == 5
    assert memory["clear_min_duration_sec"] == 2.0
    assert "memory_enabled:=true" in launcher


def test_controller_uses_one_fifteen_centimeter_arrival_window():
    config = yaml.safe_load((PACKAGE / "config/curve_tuning_waffle_pi.yaml").read_text())
    controller = config["controller_server"]["ros__parameters"]

    assert controller["general_goal_checker"]["xy_goal_tolerance"] == 0.15
    assert controller["FollowPath"]["xy_goal_tolerance"] == 0.15
    assert controller["FollowPath"]["trans_stopped_velocity"] < controller["FollowPath"]["max_vel_x"]


def test_runtime_speed_caps_match_prior_classroom_limits():
    config = yaml.safe_load((PACKAGE / "config/curve_tuning_waffle_pi.yaml").read_text())
    nodes = yaml.safe_load((PACKAGE / "config/classroom_hardware_nodes.yaml").read_text())
    controller = config["controller_server"]["ros__parameters"]["FollowPath"]
    smoother = config["velocity_smoother"]["ros__parameters"]
    mux = nodes["/motion_mux"]["ros__parameters"]
    launch = (PACKAGE / "launch/hardware.launch.py").read_text()

    assert controller["max_vel_x"] == 0.16
    assert controller["max_speed_xy"] == 0.16
    assert controller["max_vel_theta"] == 0.5
    assert smoother["max_velocity"] == [0.16, 0.0, 0.5]
    assert smoother["min_velocity"] == [-0.16, 0.0, -0.5]
    assert mux["max_linear_velocity"] == 0.16
    assert mux["max_angular_velocity"] == 0.5
    assert "parameters=[nodes, {'use_sim_time': False}]" in launch


def test_classroom_launcher_uses_current_workspace_and_stock_robot_bringup():
    launcher = (ROOT / "scripts/run_classroom_navigation_16.sh").read_text()
    tuning = (ROOT / "scripts/run_hardware_tuning.sh").read_text()
    setup = (PACKAGE / "setup.py").read_text()

    assert "LOCALIZATION_MODE=baseline" in launcher
    assert "use_motion_mux:=true" in launcher
    assert 'WORKSPACE_ROOT=$REPO_DIR/ros2_ws' in tuning
    assert "motion_mux = smart_vision_amr.motion_mux:main" in setup
