# Speed and Wall-Aware Replanning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the previous 0.16 m/s and 0.5 rad/s classroom speed limits and replan around live obstacles near mapped walls without double-inflating normal wall echoes.

**Architecture:** Classify LiDAR wall echoes in `change_memory` using the first raw static-map hit distance along each ray. The `observed_obstacles` topic then contains only dynamic hits and is consumed consistently by A* route invalidation, dynamic inflation, and memory evidence. Configure the same speed caps at the DWB, velocity-smoother, and motion-mux boundaries.

**Tech Stack:** Python 3.10, ROS 2 Humble, Nav2, pytest, colcon, YAML.

**Spec:** `docs/superpowers/specs/2026-10-29-speed-wall-aware-replanning-design.md`

## Global Constraints

- Forward and planar speed caps: 0.16 m/s; angular speed caps: 0.5 rad/s.
- Map resolution: 0.05 m/cell; wall-return range tolerance: 0.05 m.
- Persistent memory confirms after five observations spanning at least two seconds.
- Keep the 0.15 m goal tolerances, raw map unchanged, and local Nav2 collision checking.
- Do not command or move the physical robot during implementation.

## Review Focus

- A real obstacle 0.075 m in front of a mapped wall must remain a live dynamic cell and trigger a safe detour.
- A LiDAR return matching a mapped wall within 0.05 m must not create a second inflated wall that closes the corridor.
- Rotated map origins, axis-aligned rays, diagonal corner crossings, and map edges must not produce invalid ranges. Unknown cells stay blocked for A* and must not be treated as mapped walls by the LiDAR echo classifier.
- All runtime speed layers must agree, including the Nav2 smoother's previous 0.26 m/s and 1.0 rad/s defaults.
- Persistent memory must keep its five-hit / two-second confirmation and must not store expected mapped-wall echoes.

---

### Task 1: Classify mapped-wall returns by measured range

**Files:**
- Modify: `ros2_ws/src/smart_vision_amr/smart_vision_amr/map_utils.py`
- Modify: `ros2_ws/src/smart_vision_amr/smart_vision_amr/change_memory.py`
- Modify: `ros2_ws/src/smart_vision_amr/smart_vision_amr/astar_planner.py`
- Modify: `ros2_ws/src/smart_vision_amr/config/classroom_hardware_nodes.yaml`
- Test: `tests/test_live_obstacles.py`
- Test: `tests/test_planner_recovery.py`
- Test: new `tests/test_wall_return_classification.py`

**Interfaces:**
- Produces: `first_static_obstacle_distance(info, grid, origin_x, origin_y, angle, max_distance) -> float | None`, where `grid` is the uninflated static `Grid`; `None` means no mapped wall is intersected in range.
- Produces: `/smart_vision_amr/observed_obstacles` contains dynamic LiDAR endpoints only. Both A* planning and route-intersection checks consume those cells without a second geometric filter. Statically free dynamic cells can enter persistent memory; raw-static occupied cells stay transient because free-space evidence cannot clear them.

- [x] **Step 1: Write failing ray-geometry and scan-classification tests**

Test exact first-wall range for axis-aligned and rotated map origins, a normal wall echo within 0.05 m, and a real endpoint 0.075 m before that wall. Assert that only the real endpoint is published and eligible for memory.

- [x] **Step 2: Run focused tests to verify the regression fails**

Run: `PYTHONPATH=ros2_ws/src/smart_vision_amr:$PYTHONPATH python3 -m pytest -q tests/test_wall_return_classification.py tests/test_live_obstacles.py`
Expected: FAIL because current wall classification has no range-aware ray API and current observed cells use cell-proximity filtering downstream.

- [x] **Step 3: Implement map ray traversal and producer-side classification**

Cache the raw map grid when the map changes. Traverse each valid scan ray through the map grid, suppress only hits within the configured one-cell range tolerance, and publish/store dynamic endpoints. Keep dynamic live cells even if a measured endpoint falls in an occupied map cell, but only add cells that are statically free to persistent memory.

- [x] **Step 4: Make A* consume the classified live-obstacle set directly**

Remove the cell-distance wall filter from `astar_planner.py`. Use the same dynamic cells in route intersection, clearance costs, inflation, and no-path diagnostics. Update or replace tests that asserted post-hoc wall filtering.

- [x] **Step 5: Verify a near-wall dynamic hit triggers an A* detour**

Feed the classified hit into the planner with an active path along a wall-adjacent corridor. Assert that the current path is invalidated immediately and the retained goal gets a replacement path outside the dynamic inflation region after the retry interval.

- [x] **Step 6: Run focused navigation tests**

Run: `PYTHONPATH=ros2_ws/src/smart_vision_amr:$PYTHONPATH python3 -m pytest -q tests/test_wall_return_classification.py tests/test_live_obstacles.py tests/test_planner_recovery.py`
Expected: PASS, including the normal wall-echo corridor case and the wall-adjacent obstacle detour.

### Task 2: Align runtime speed caps and project documentation

**Files:**
- Modify: `ros2_ws/src/smart_vision_amr/config/curve_tuning_waffle_pi.yaml`
- Modify: `ros2_ws/src/smart_vision_amr/config/classroom_hardware_nodes.yaml`
- Modify: `ros2_ws/src/smart_vision_amr/launch/hardware.launch.py`
- Modify: `ros2_ws/src/smart_vision_amr/smart_vision_amr/motion_mux.py`
- Modify: `README.md`
- Modify: `docs/HANDOFF_2026-10-26.md`
- Test: `tests/test_navigation_profile.py`
- Test: `tests/test_motion_mux.py`

**Interfaces:**
- DWB `max_vel_x` and `max_speed_xy`: `0.16`; DWB `max_vel_theta`: `0.5`.
- `velocity_smoother.max_velocity`: `[0.16, 0.0, 0.5]`; `min_velocity`: `[-0.16, 0.0, -0.5]`.
- Motion mux defaults and classroom node parameters: `max_linear_velocity=0.16`, `max_angular_velocity=0.5`.

- [x] **Step 1: Write failing configuration and motion-mux cap tests**

Assert exact DWB, velocity-smoother, classroom motion-mux, and `MotionMuxCore` default caps; assert an over-limit command is clamped to 0.16 m/s and 0.5 rad/s.

- [x] **Step 2: Run tests to verify they fail on existing values**

Run: `PYTHONPATH=ros2_ws/src/smart_vision_amr:$PYTHONPATH python3 -m pytest -q tests/test_navigation_profile.py tests/test_motion_mux.py`
Expected: FAIL because DWB is 0.12/1.0, velocity smoother inherits 0.26/1.0, and the motion mux defaults to 0.25/1.0.

- [x] **Step 3: Align DWB, smoother, and motion-mux settings**

Set the caps in the package tuning YAML, classroom node parameters, launch wiring, and motion-mux defaults. Keep acceleration, goal, footprint, and other planner settings unchanged.

- [x] **Step 4: Update README and handoff speed descriptions**

Document 0.16 m/s and 0.5 rad/s as the effective caps across all three runtime layers.

- [x] **Step 5: Run the full suite and build**

Run: `PYTHONPATH=ros2_ws/src/smart_vision_amr:$PYTHONPATH python3 -m pytest -q tests`
Expected: all tests pass.

Run: `source /opt/ros/humble/setup.bash && colcon build --base-paths ros2_ws/src --packages-select smart_vision_amr --symlink-install`
Expected: package builds successfully.

- [x] **Step 6: Inspect and commit the completed change**

Run `git diff --check`, inspect the final diff and `git status`, then commit in English with a message describing wall-aware obstacle replanning and restored speed caps.
