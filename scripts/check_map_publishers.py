#!/usr/bin/env python3
"""Refuse hardware Nav2 startup while another map source owns ROS /map.

Query DDS directly: the ros2 daemon can retain a stale graph after a previous
SLAM session, and topic-info text differs when /map does not exist yet.
"""

import argparse
import sys
import time

import rclpy


SLAM_NODES = {'cartographer_node', 'cartographer_occupancy_grid_node', 'slam_toolbox'}


def full_name(namespace, name):
    return f"{namespace.rstrip('/')}/{name}" if name else '<unknown /map publisher>'


def conflicting_sources(node):
    """Return live map publishers and SLAM nodes that may soon publish /map."""
    publishers = {
        full_name(info.node_namespace, info.node_name)
        for info in node.get_publishers_info_by_topic('/map')
    }
    mapping_nodes = {
        full_name(namespace, name)
        for name, namespace in node.get_node_names_and_namespaces()
        if name in SLAM_NODES
    }
    return sorted(publishers | mapping_nodes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain-id', required=True)
    parser.add_argument('--discovery-seconds', type=float, default=2.5)
    options = parser.parse_args(argv)
    if not 0 <= options.discovery_seconds <= 10:
        parser.error('--discovery-seconds must be between 0 and 10')
    node = None
    try:
        rclpy.init()
        node = rclpy.create_node('smart_vision_map_source_preflight')
        deadline = time.monotonic() + options.discovery_seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=max(0.0, min(.25, deadline-time.monotonic())))
        sources = conflicting_sources(node)
    except Exception as error:
        print(f'ROS 도메인 {options.domain_id}의 /map 발행자를 확인하지 못했습니다: {error}',
              file=sys.stderr)
        return 2
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if sources:
        print(f'ROS 도메인 {options.domain_id}에 기존 /map 발행자 또는 SLAM 노드가 있습니다: '
              + ', '.join(sources), file=sys.stderr)
        print('저장 지도 주행 전 SLAM/Cartographer와 기존 지도 서버를 종료하세요. '
              '동시에 두 지도를 발행하면 AMCL과 경로 계획의 좌표가 섞입니다.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
