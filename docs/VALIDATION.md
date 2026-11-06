# 추출본 검증 기록

검증일: 2026-11-06 (한국 시간).
환경: macOS, Python 3.14, pytest, NumPy, OpenCV. ROS 2 및 실제 로봇 연결 없음.

## 실행 결과

- `python -m pytest -q`: **274 passed**, ROS 없는 환경에서 선별 테스트 16개 파일 실행.
- `python setup.py build_ext --inplace`: 선택적 C++ 기하 확장 빌드 성공 (macOS clang++).
- 원본 보존 파일: `source_manifest.json`에 기록한 SHA-256과 추출 파일 대조.
- 패키지 내부 import 의존성, ROS console entry point, 로컬 문서 링크 확인.
- `git diff --check`: 공백 오류 검사.

원본 테스트 중 ROS 노드·메시지 또는 원본 전체 파일 구성이 필요한 항목은 독립 실행
테스트 묶음에 포함하지 않았습니다. `test_command_ramp.py`의 MUX 출력 검증 하나는
다른 테스트가 주입한 ROS 메시지 대역에 의존했습니다. 해당 테스트 내부에서
`monkeypatch`로 Twist/String 대역을 제공하도록 바꾸고 원래 assert는 유지했습니다.
이 테스트 파일은 수정한 항목으로 별도 표시했습니다.

## 검증하지 않은 범위

ROS 2 Humble의 Python 3.10 환경, colcon 전체 빌드, DWB 플러그인의 ROS 빌드,
Nav2 action 통합, 실차 주행, 실제 제동 성능 및 반복 임무 성공률은 이번 작업에서
검증하지 않았습니다. 단위 테스트 결과를 통합 주행 성공률로 해석하지 않습니다.
