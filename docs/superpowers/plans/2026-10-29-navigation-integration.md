# Navigation Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate teammate classroom navigation with persistent memory, consistent arrival tolerance, and recovery from stale navigation inputs.

**Architecture:** Import the teammate's tested navigation modules and classroom parameters without web/camera/mission subsystems. Keep the path dense; a motion mux owns `/cmd_vel`. Planner retains goals during temporary failure. A follower input guard cancels stale-input motion and relies on the planner's bounded retry after recovery.

**Tech Stack:** Python 3.10, ROS 2 Humble, Nav2, pytest, colcon.

**Spec:** `docs/superpowers/specs/2026-10-29-navigation-integration-design.md`

## Global Constraints

- Domain 16, stock robot bringup, real time; no robot movement during implementation.
- Memory: five occupied observations and at least two seconds; five free observations and at least two seconds to clear.
- Align goal checker and DWB xy tolerances at 0.15 m.
- Manual stop and emergency stop remain authoritative.

## Review Focus

- A LiDAR return near a mapped wall must not erase a real new obstacle beside that wall.
- Transient or stale sensor input must stop a live action and later permit a new plan for the same goal.
- A queued path from before manual stop must not resume motion.
- Motion mux must be the only `/cmd_vel` publisher in the provided launch.
- A robot starting inside an inflated halo must not be allowed through a physically occupied footprint.

---

### Task 1: Import the navigation base

**Files:** `ros2_ws/src/smart_vision_amr/smart_vision_amr/*.py`, `config/*.yaml`, `maps/*`, `launch/hardware.launch.py`, `setup.py`, `package.xml`, `scripts/*`, `tests/*`.

- [x] Add a focused test for the imported wider-route A* and verified reverse selection; observe it fail against the checkpoint.
- [x] Import teammate navigation modules and their focused unit tests, keeping unrelated web/camera modules out.
- [x] Provide baseline classroom launch with motion mux and duplicate-stack preflight.
- [x] Run navigation test suite and build the ROS package.

### Task 2: Preserve memory and arrival behavior

**Files:** `launch/hardware.launch.py`, `config/curve_tuning_waffle_pi.yaml`, `scripts/run_classroom_navigation_16.sh`, memory tests.

- [x] Add integration assertions for memory defaults, five observations/two seconds, and both 0.15 m arrival tolerances; observe expected failure.
- [x] Enable classroom persisted memory by default and align both tolerances.
- [x] Run focused tests.

### Task 3: Guard live navigation inputs

**Files:** `smart_vision_amr/path_follower.py`, `smart_vision_amr/navigation_inputs.py`, follower tests.

- [x] Add tests for stale scan/odom/costmap/footprint refusal, active cancellation, and successful retry after input recovery; observe failure.
- [x] Implement the guard with publisher-compatible QoS and integrate with the follower's existing goal/invalid-path state machine.
- [x] Run the complete suite and colcon build, inspect diff, then commit in English.

### Task 4: Address review findings

- [x] Keep full robot clearance around confirmed wall-adjacent memory; document the wall-echo ambiguity and safe clearing procedure.
- [x] Require the circular robot footprint to clear raw obstacle squares throughout dynamic-halo egress.
- [x] Add regressions for corridor retention, actual obstacle blocking, and physically clear egress.
