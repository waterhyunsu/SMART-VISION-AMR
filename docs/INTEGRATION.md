# 주행 통합 메모

이 추출본은 팀 최종본의 알고리즘과 직접 의존성을 보존합니다. 원본 Python/C++ 구현과
수정하지 않은 선별 테스트·설정은 `source_manifest.json`의 SHA-256으로 비교할 수 있습니다.
패키지 메타데이터, 문서, 테스트 실행 설정은 개인 저장소 범위에 맞춰 정리했습니다.

## 포함 경계

`astar_planner`, `path_follower`, `change_memory`가 알고리즘의 ROS 연결을 맡습니다.
`motion_mux`, `command_stamper`, `localization_health`는 최종 주행 연결을 위한 지원 코드입니다.
`ekf_sensor_guard`는 내부 참조를 충족하기 위해 포함했으며 EKF 런치를 제공하지 않습니다.

## 외부 실행 환경

- TurtleBot3 bringup과 드라이버, 지도 서버, AMCL, TF 트리.
- Nav2 controller / FollowPath, costmap, velocity smoother 및 lifecycle 관리.
- Nav2 속도 제한 입력 `/speed_limit` 연결.
- 최종 Nav2 출력과 behavior 출력의 `/smart_vision_amr/navigation_cmd_vel` 연결.
- `/cmd_vel` 최종 발행자는 중복되지 않게 구성.

`classroom_demo_operator_20261104.yaml`에는 팀의 별도 C++ 패키지
`smart_vision_amr_dwb` 플러그인 설정이 포함되어 있습니다. 해당 패키지는 최종 주행 설정의 직접 의존성이므로 원본 그대로 포함했습니다.
워크스페이스 전체를 colcon으로 빌드하면 함께 빌드됩니다.
기존 Nav2 controller 설정을 사용할 때는 FollowPath controller ID와 goal checker ID도
`path_follower` 설정에 맞춰야 합니다. 설정 파일은 전체 bringup의 대체물이 아닙니다.

## 비교와 검증

원본 구현을 기능 개선 목적으로 다시 작성하지 않았습니다. 이번 커밋의 목적은 최신 팀 코드에서
개인 담당 알고리즘을 독립적으로 읽고 검증할 수 있는 형태로 추출하는 것입니다.
단위 테스트 통과는 ROS 2 통합, 하드웨어 연결 또는 실제 주행 성공의 증거를 대신하지 않습니다.
