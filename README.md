# SMART-VISION-AMR — TurtleBot3 Waffle Pi 실기 A* 주행

Ubuntu 22.04 / ROS 2 Humble용 실기 주행 패키지입니다. Raspberry Pi의 TurtleBot3 bringup과
Linux PC의 Nav2를 사용합니다. 기본 지도는 `classroom_20261027_cleaned.yaml`입니다.
Gazebo, 카메라, ArUco 기능은 포함하지 않습니다.

이전 작업 기록은 [2026-10-26 핸드오프](docs/HANDOFF_2026-10-26.md)에 있습니다.
이 문서의 실행 명령과 설정은 이후의 주행 통합 버전을 기준으로 합니다.

## 실행 명령 한눈에 보기

최초 설치와 빌드를 마친 후, **Pi 터미널**과 **Linux PC 터미널**에서 각각 실행합니다.

```bash
# Raspberry Pi: 로봇 센서와 모터
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash  # 해당 워크스페이스에 설치한 경우
export TURTLEBOT3_MODEL=waffle_pi LDS_MODEL=LDS-03 ROS_DOMAIN_ID=16
ros2 launch turtlebot3_bringup robot.launch.py use_sim_time:=false
```

```bash
# Linux PC: 중복 실행 검사 후 지도, Nav2, A*, 주행 명령 중재
source /opt/ros/humble/setup.bash
cd ~/SMART-VISION-AMR/ros2_ws
colcon build --symlink-install --packages-select smart_vision_amr
../scripts/run_classroom_navigation_16.sh
```

Pi에서 이미 bringup이 실행 중이면 PC 명령만 실행합니다. 설치 위치가 다르면
`cd` 경로를 바꾸세요. 이 스크립트는 도메인 16, 실제 시간, 저장 지도,
AMCL, 동작 명령 중재 및 장애물 기억을 사용하고 RViz는 열지 않습니다.
별도 터미널에서 RViz를 열고 **2D Pose Estimate**로 시작 위치를 잡은 뒤
**2D Goal Pose**로 도착지를 지정합니다.

## 구성

`/smart_vision_amr/goal` → 여유 거리 비용을 사용하는 A* (`/smart_vision_amr/path`)
→ Nav2 `/follow_path` → `/smart_vision_amr/navigation_cmd_vel`
→ motion mux `/cmd_vel` → TurtleBot3입니다. `/scan` 관측은 현재 장애물 회피에
즉시 사용하고, 확인된 장애물은 지도와 별도 파일에 기억합니다. 경로가 막히면
목표를 유지한 채 재탐색하고, 뒤쪽 안전 조건이 만족되면 제한적으로 후진합니다.

> **적응형 장애물 기억 기준:** 같은 지도 셀이 장애물로 **5회 이상 관측되고,
> 첫 관측부터 최소 2초가 지난 경우에만** 기억 셀로 확정합니다. 두 조건을
> 모두 만족해야 합니다. 확정된 셀은 A*가 장애물로 취급하고, 변경된 셀이 현재
> 경로를 막으면 새 경로를 계산합니다. 기억은 기본적으로 켜져 있으며,
> `~/.ros/smart_vision_amr/classroom_memory.json`에 저장됩니다. 현재 관측은
> 기억 확정 전에도 경로 계획에 반영됩니다.
> 셀이 비어 있는 것으로 5회 이상, 최소 2초간 관측되면 기억에서 해제합니다.

현재 라이다가 지도 벽에서 2셀(10cm) 이내로 잡힌 경우, A*는 지도 벽의
반사 위치 오차로 보고 현재 관측의 이중 팽창을 피합니다. 다만 **확정된 기억
셀에는 전체 5셀(25cm) 안전 반경을 적용합니다.** 벽 가까운 실제 장애물이
가려졌을 때도 안전거리를 유지하기 위해서입니다. 지도 벽의 반사 오차가
반복되어 기억 셀로 확정되면 좁은 통로의 경로가 막힐 수 있습니다. 그때는
지도·위치 추정·라이다를 확인하고, 잘못된 기억임을 확인한 경우에만
`ros2 service call /smart_vision_amr/clear_memory std_srvs/srv/Trigger '{}'`로
기억을 지우세요.

## 1. 로봇 Raspberry Pi

이미 하드웨어 bringup과 PC 간 ROS 토픽 공유가 된 환경에서 실행합니다.
Pi의 `turtlebot3_bringup`과 LDS-03용 `coin_d4_driver`가 설치돼 있어야 합니다.
없다면 [ROBOTIS의 Ubuntu 22.04/Humble SBC 설치 안내](https://emanual.robotis.com/docs/en/platform/turtlebot3/sbc_setup/)에
따라 Pi에 TurtleBot3 `humble` 브랜치와 `coin_d4_driver`를 설치합니다.
PC와 Pi는 같은 네트워크 및 같은 `ROS_DOMAIN_ID`를 사용하세요.

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash  # 이 경로에 ROBOTIS 패키지를 빌드한 경우
export TURTLEBOT3_MODEL=waffle_pi
export LDS_MODEL=LDS-03
export ROS_DOMAIN_ID=16  # PC에도 같은 값을 설정; 이미 다른 값을 쓰고 있다면 그 값 유지
ros2 pkg prefix turtlebot3_bringup
ros2 pkg prefix coin_d4_driver
ros2 launch turtlebot3_bringup robot.launch.py use_sim_time:=false
```

Pi에서 `/scan`, `/odom`, `/tf`, `/cmd_vel`이 존재해야 합니다. LDS-03 드라이버는
설치된 ROBOTIS TurtleBot3 bringup의 `LDS_MODEL=LDS-03` 지원을 사용합니다.
기본 장치 경로는 OpenCR `/dev/ttyACM0`, LDS-03 `/dev/ttyUSB0`입니다.
장치 연결이 다르면 포트와 권한을 확인하세요. ROS 패키지가 이미 시스템에
설치되어 `~/turtlebot3_ws`가 없다면 위 `source` 줄은 생략합니다.

## 2. Linux PC 설치 및 실행

PC와 Pi에서 같은 `ROS_DOMAIN_ID`를 사용합니다. PC에 ROS 2 Humble, Nav2,
TurtleBot3 navigation 패키지가 설치되어 있어야 합니다.

```bash
sudo apt update
sudo apt install ros-humble-desktop ros-humble-navigation2 ros-humble-nav2-bringup \
  ros-humble-turtlebot3-navigation2 ros-humble-rviz2 \
  python3-colcon-common-extensions python3-rosdep
git clone https://github.com/waterhyunsu/SMART-VISION-AMR.git
cd SMART-VISION-AMR/ros2_ws
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=16  # Pi와 동일한 값
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
export TURTLEBOT3_MODEL=waffle_pi
python3 ../scripts/install_tf2_02520_overlay.py  # tf2 0.25.21~0.25.24인 PC에서 1회
../scripts/run_classroom_navigation_16.sh
```

Humble의 `tf2` 0.25.23에서 Nav2 TF 수신이 멈추고, 로컬 코스트맵은 오래된
로봇 자세를 사용한 채 새 타임스탬프로 발행되는 현상을 확인했습니다. 위 스크립트는
ROS 공식 2026-07-02 스냅샷의 서명과 패키지 해시를 검증한 뒤 정상 동작이
보고된 `tf2` 0.25.20을 사용자 디렉터리에 설치합니다. 시스템 apt 패키지는
수정하지 않습니다. 해당 버전의 `tf2`가 설치된 PC에서는 `hardware.launch.py`가
이 로컬 버전을 자동으로 사용하고, 로컬 버전이 없으면 주행 런치를 중단합니다.
Nav2는 기본적으로 노드를 개별 프로세스로 실행합니다.
현재 교실 설정의 DWB 상한은 전진 0.12m/s, 회전 1.0rad/s입니다.
위치 도착 허용 오차와 최종 회전 진입 거리 기준은 모두 0.15m입니다.

기본 교실 실행 스크립트는 RViz를 열지 않습니다. 별도 PC 터미널에서
`ros2 run rviz2 rviz2 -d ~/SMART-VISION-AMR/ros2_ws/src/smart_vision_amr/rviz/minimal.rviz`
로 열 수 있습니다. 직접 `ros2 launch smart_vision_amr hardware.launch.py`를
실행하면 RViz도 열리지만, 중복 실행 검사를 포함한 스크립트 사용을 권장합니다.

기본 Nav2 설정은 저장소의 `config/curve_tuning_waffle_pi.yaml`을 사용합니다.
지도는 정리된 교실 지도입니다. 실제 지도와 로봇 위치가
맞아야 합니다. 다른 지도를 사용하려면 `map:=/절대경로/map.yaml`을 지정합니다.

## 3. 주행 전 확인

로봇을 넓고 비어 있는 곳에 놓고 다음을 확인합니다.

```bash
ros2 topic hz /scan
ros2 topic echo /odom --once
ros2 topic echo /map --once
ros2 run tf2_ros tf2_echo odom base_footprint
ros2 action list | grep follow_path
ros2 lifecycle get /controller_server
```

별도 터미널에서 연 RViz에서 실제 위치에 맞춰 **2D Pose Estimate**를 지정한 뒤
`ros2 run tf2_ros tf2_echo map base_footprint`가 연속적으로 출력되는지
확인합니다. 지도와 위치 추정이 맞지 않으면 목표를 보내지 마세요.
또한 `/local_costmap/published_footprint`의 `header.stamp`가 현재 시간에 맞게
계속 증가해야 합니다. 오래된 코스트맵, `/scan`, `/odom`이 감지되면
`path_follower`가 목표 전송을 거부하거나 진행 중인 목표를 취소합니다.

## 4. 첫 목표

RViz 상단의 **2D Goal Pose**를 선택해 지도 안의 가깝고 비어 있는 지점을
클릭하고 방향으로 드래그합니다. 이 버튼은 `/smart_vision_amr/goal`로
직접 발행하도록 설정되어 있어 A* 경로가 사용됩니다. 지도 위의 시작 위치는
**2D Pose Estimate**로 지정합니다. PC의 launch를 실행해도 Pi에서 로봇
bringup은 별도로 실행되어 있어야 합니다.

클릭 대신 터미널로 목표를 보낼 수도 있습니다. 아래 좌표는 실제 지도에서
확인한 자유 공간 좌표로 바꿔야 합니다.

```bash
ros2 topic pub --once /smart_vision_amr/goal geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: map}, pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}'
ros2 topic echo /smart_vision_amr/navigation_status
ros2 topic echo /cmd_vel
```

중단은 `ros2 service call /smart_vision_amr/cancel_navigation std_srvs/srv/Trigger '{}'`입니다.
첫 주행은 바퀴를 띄워 방향을 확인하고, 이후 넓은 공간에서 저속으로 수행하세요.
팀원 코드의 Sector 3 → Sector 2 → Home 실차 성공 기록이 있습니다. 이 저장소에서
통합한 버전은 코드 시험과 빌드를 거쳤으며 실차 통합 주행 확인은 남아 있습니다.

## 문제를 좁히는 순서

1. `/map`이 없으면 map server와 지도 경로를 확인합니다.
2. `map -> base_footprint` TF가 없으면 AMCL 초기 위치 및 Pi/PC 시계 동기화를 확인합니다.
3. `/smart_vision_amr/path`가 비면 A* 로그의 시작/목표 셀과 지도 장애물 판정을 확인합니다.
4. FollowPath가 실패하면 controller 로그, local costmap, `/scan`을 확인합니다.
5. `/cmd_vel`이 나오는데 바퀴가 돌지 않으면 Pi의 OpenCR/모터 쪽을 확인합니다.

`config/classroom_hardware_nodes.yaml`의 `obstacle_inflation_cells`는 0.05 m 해상도 기준
5셀(0.25 m)입니다. 실제 통로 폭과 로봇 외형을 확인한 뒤 조정하세요.
Nav2의 controller/local costmap은 저장소의 교실 튜닝 설정을 따릅니다.

## 출처

A* 격자 탐색, 장애물 관측/기억 및 FollowPath 연결은
[`resilient-amr-astar`](https://github.com/waterhyunsu/resilient-amr-astar)의
구현을 참고해 실기 전용 패키지로 구성했습니다.
