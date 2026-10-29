# Navigation integration design

## Intent

Use the teammate's classroom navigation implementation as the navigation base while retaining the project's persistent obstacle memory and practical safety checks. The robot must keep its explicit destination through temporary blockage and sensor loss, replan around new obstacles, and use bounded reverse recovery only when its rear corridor is verified. Keep the camera and web mission separate from the navigation package change.

## Approach

Import the teammate's A* planner, follower, motion arbiter, live obstacle overlay, classroom map and tuning as a coherent navigation stack. Keep dense grid paths until both execution and collision validation use the same geometry. Turn persisted memory on by default in the classroom launcher, retain the existing five-observation/two-second confirmation and clearing rules, and align Nav2's goal checker with DWB at 0.15 m. Add a small normal-navigation sensor guard that cancels stale-input motion and allows the retained goal to retry when inputs recover.

The alternatives were to patch the original checkpoint in place or to copy the entire teammate project including its camera/web/simulation stack. The former leaves the tested recovery integration behind; the latter changes unrelated systems. This scoped import follows the approved comparison and integration direction.

## Constraints and proof

- ROS 2 Humble on a physical TurtleBot3 Waffle Pi, domain 16, real time.
- One map publisher and one automatic motion owner per domain.
- Manual stop and emergency stop must not be undone by automatic retry.
- Remembered cells are persisted only after at least five observations over at least two seconds. Transient lidar obstacles influence planning immediately.
- The classroom path must run with stock robot bringup in baseline localization mode. EKF remains outside this integration.
- Verify source-level behavior with unit tests and a colcon build. Physical route stability requires a later robot run and is not asserted by this commit.

## Review resolutions

- A confirmed remembered LiDAR endpoint always receives the full obstacle inflation, including near mapped walls. A near-wall hit can be either a wall echo or a real object; preserving real-obstacle clearance takes priority. The transient layer still filters wall echoes to avoid immediate double inflation. False confirmed wall echoes may close a narrow corridor until they are observed free or explicitly cleared after checking the map and localization.
- The exceptional A* start escape from dynamic inflation is allowed only if every nominated cell clears raw static and dynamic obstacle squares, the map edge, and the circular robot footprint plus margin. A physically occupied start receives no escape route.
