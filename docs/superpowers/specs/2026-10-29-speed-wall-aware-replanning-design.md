# Restore classroom speed limits and make wall-adjacent obstacles replan

## Intent

Restore the previously used Waffle Pi limits (0.16 m/s forward and 0.5 rad/s turning) consistently across DWB, Nav2's velocity smoother, and the motion mux. Make a live obstacle close to a mapped wall reach A* immediately, so Nav2 stopping locally does not leave the global planner following the same blocked route.

## Root cause and chosen approach

The classroom planner currently treats any observed cell within two map cells (0.10 m at the classroom map's 0.05 m resolution) of a mapped wall as a wall return. This decision uses only endpoint-cell distance, not the measured laser range. A real object in that band can therefore be absent from A*'s live obstacle set while Nav2's local costmap still sees it and stops.

Keep narrow-corridor protection, but classify wall echoes at the scan source: ray-trace each valid beam through known occupied cells in the raw map, compare its measured hit distance with the first mapped occupied-cell distance, and suppress it only when the two agree within one map cell (0.05 m). Unknown cells remain blocked for A* but are not evidence of a mapped wall. A closer return is a live obstacle even when its endpoint lies near or on a mapped wall. Publish only classified dynamic cells on `observed_obstacles`; use those cells for route invalidation and A* inflation. Feed statically free dynamic endpoints into persistent memory; skip cells already occupied in the raw map because free-space evidence cannot clear them. Persistent memory still confirms only after five observations spanning at least two seconds.

Alternatives considered: removing the wall filter entirely would double-inflate mapped wall echoes and close narrow passages; retaining cell-only filtering would preserve the current blind spot. Range-aware classification addresses both while keeping the existing footprint inflation and memory thresholds.

## Constraints and proof

- Use the known previous limits: DWB x and planar speed 0.16 m/s; DWB angular speed 0.5 rad/s; velocity-smoother and motion-mux caps must not exceed those values.
- Preserve 0.15 m goal tolerances, five observations / two seconds for persistent memory, source-map immutability, Nav2 local collision checking, and real-time operation.
- Ignore a mapped-wall return only when measured range agrees with the first raw map obstacle within 0.05 m. A return more than 0.05 m closer remains dynamic and can trigger a global replan immediately.
- Test the wall echo and a nearer obstacle in the same corridor geometry, and test that the latter produces a safe A* detour. Run the full Python suite and ROS package build. Physical robot behavior remains subject to a later hardware test.
