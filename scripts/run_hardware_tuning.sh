#!/usr/bin/env bash
# PC-side real-robot navigation; run TurtleBot3 bringup on the robot separately.
set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
WORKSPACE_ROOT=$REPO_DIR/ros2_ws
PARAMS_FILE=${TURTLEBOT3_TUNING_PARAMS_FILE:-$REPO_DIR/ros2_ws/src/smart_vision_amr/config/curve_tuning_waffle_pi.yaml}
DOMAIN_ID=${TURTLEBOT3_TUNING_DOMAIN_ID:-16}
if [[ $# == 0 || "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: run_hardware_tuning.sh /absolute/path/real_robot_map.yaml [LAUNCH_ARGUMENT:=VALUE ...]
PC-side AMCL/Nav2/A* with the editable curve_tuning_waffle_pi.yaml preset.
Default domain: 16; real time; RViz disabled. Robot bringup runs separately.
TURTLEBOT3_TUNING_DOMAIN_ID and TURTLEBOT3_TUNING_PARAMS_FILE override defaults.
EOF
  if [[ $# == 0 ]]; then exit 2; else exit 0; fi
fi
MAP_FILE=$(realpath -- "$1")
shift
if [[ ! -f "$MAP_FILE" || ! -f "$PARAMS_FILE" || ! -f "$WORKSPACE_ROOT/install/local_setup.bash" ]]; then
  printf 'Map, tuning YAML, or built SMART-VISION-AMR workspace is missing.\n' >&2
  exit 2
fi
if [[ ! "$DOMAIN_ID" =~ ^[0-9]+$ || "$DOMAIN_ID" -gt 232 ]]; then
  printf 'TURTLEBOT3_TUNING_DOMAIN_ID must be 0..232.\n' >&2
  exit 2
fi
# One physical navigation stack per ROS domain: a second launch would create
# duplicate map_server/AMCL/lifecycle-manager names and break their bonds.
LOCK_DIR=${XDG_RUNTIME_DIR:-$HOME/.cache}/smart-vision-amr
mkdir -p -- "$LOCK_DIR"
chmod 700 -- "$LOCK_DIR"
exec 9>"$LOCK_DIR/hardware-navigation-${DOMAIN_ID}.lock"
if ! flock -n 9; then
  printf 'Navigation is already running in ROS domain %s. Stop that launch before starting another.\n' "$DOMAIN_ID" >&2
  exit 1
fi
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH ROS_PACKAGE_PATH PYTHONPATH LD_LIBRARY_PATH
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$WORKSPACE_ROOT/install/local_setup.bash"
export ROS_DOMAIN_ID="$DOMAIN_ID" ROS_LOCALHOST_ONLY=0 TURTLEBOT3_MODEL=waffle_pi
# Also catch a Nav2 stack started directly, outside this script/lock. Query the
# live graph rather than the ROS daemon cache, which can retain old node names.
if ! EXISTING_NODES=$(timeout 10s ros2 node list --no-daemon --spin-time 2); then
  printf 'Could not inspect ROS domain %s; refusing to risk a duplicate navigation launch.\n' "$DOMAIN_ID" >&2
  exit 1
fi
if printf '%s\n' "$EXISTING_NODES" | grep -Eq '^/(map_server|amcl|lifecycle_manager_localization)$'; then
  printf 'A map server, AMCL, or localization lifecycle manager already exists in ROS domain %s. Stop the old navigation launch first.\n' "$DOMAIN_ID" >&2
  exit 1
fi
# A lingering Cartographer or other /map publisher can overwrite Nav2's saved
# map for AMCL, the A* planner, and the web UI. Check the live DDS graph rather
# than the ros2 daemon's cache, before starting another map_server.
if ! timeout 10s /usr/bin/python3 "$SCRIPT_DIR/check_map_publishers.py" --domain-id "$DOMAIN_ID"; then
  printf 'Refusing to start hardware navigation until ROS domain %s has one planned map source.\n' "$DOMAIN_ID" >&2
  exit 1
fi
exec ros2 launch smart_vision_amr hardware.launch.py \
  map:="$MAP_FILE" params_file:="$(realpath -- "$PARAMS_FILE")" use_rviz:=false "$@"
