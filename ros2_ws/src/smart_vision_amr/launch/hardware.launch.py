"""PC-side navigation. Start turtlebot3_bringup robot.launch.py on the Pi separately."""

import os
from pathlib import Path
from xml.etree import ElementTree

from nav2_common.launch import RewrittenYaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _tf2_environment_actions():
    manifest = Path('/opt/ros/humble/share/tf2/package.xml')
    if not manifest.exists():
        return []
    version = ElementTree.parse(manifest).getroot().findtext('version', default='0.0.0')
    numbers = tuple(int(part) for part in version.split('.')[:3])
    if not (0, 25, 21) <= numbers <= (0, 25, 24):
        return []
    prefix = (Path.home() / '.local/share/smart_vision_amr/tf2-0.25.20/root/opt/ros/humble')
    if not (prefix / 'lib/libtf2.so').exists() or not (prefix / 'lib/libtf2_ros.so').exists():
        raise RuntimeError(
            f'tf2 {version} can freeze Nav2 TF updates. Install the verified tf2 0.25.20 '
            f'overlay at {prefix} before starting hardware navigation.')
    libraries = str(prefix / 'lib')
    python_paths = [str(prefix / 'lib/python3.10/site-packages'),
                    str(prefix / 'local/lib/python3.10/dist-packages')]
    return [
        SetEnvironmentVariable('LD_LIBRARY_PATH',
                               os.pathsep.join(part for part in
                                               [libraries, os.environ.get('LD_LIBRARY_PATH')]
                                               if part)),
        SetEnvironmentVariable('PYTHONPATH',
                               os.pathsep.join(part for part in
                                               python_paths + [os.environ.get('PYTHONPATH')]
                                               if part)),
    ]


def generate_launch_description():
    own = FindPackageShare('smart_vision_amr')
    nav2 = FindPackageShare('nav2_bringup')
    tb3 = FindPackageShare('turtlebot3_navigation2')
    map_default = PathJoinSubstitution([own, 'maps', 'classroom_map.yaml'])
    params_default = PathJoinSubstitution([tb3, 'param', 'humble', 'waffle_pi.yaml'])
    nodes = PathJoinSubstitution([own, 'config', 'nodes.yaml'])
    rviz_config = PathJoinSubstitution([own, 'rviz', 'minimal.rviz'])
    memory_default = '~/.ros/smart_vision_amr/classroom_memory.json'
    safe_nav_params = RewrittenYaml(
        source_file=LaunchConfiguration('params_file'), root_key='', convert_types=True,
        param_rewrites={
            'amcl.ros__parameters.transform_tolerance': '0.1',
            'local_costmap.local_costmap.ros__parameters.robot_radius': '0.20',
            'local_costmap.local_costmap.ros__parameters.inflation_layer.inflation_radius': '0.25',
            'local_costmap.local_costmap.ros__parameters.publish_frequency': '5.0',
            'global_costmap.global_costmap.ros__parameters.robot_radius': '0.20',
            'global_costmap.global_costmap.ros__parameters.inflation_layer.inflation_radius': '0.25',
            'controller_server.ros__parameters.FollowPath.max_vel_x': '0.16',
            'controller_server.ros__parameters.FollowPath.max_speed_xy': '0.16',
            'controller_server.ros__parameters.FollowPath.vx_samples': '4',
            'controller_server.ros__parameters.FollowPath.max_vel_theta': '0.5',
            'controller_server.ros__parameters.FollowPath.trans_stopped_velocity': '0.02',
            # Keep DWB's rotate-to-goal trigger aligned with Nav2's checker.
            # Together with A*'s 0.05 m endpoint snap, this limits early
            # success at a position noticeably short of the requested goal.
            'controller_server.ros__parameters.general_goal_checker.xy_goal_tolerance': '0.15',
            'controller_server.ros__parameters.FollowPath.xy_goal_tolerance': '0.15',
            'velocity_smoother.ros__parameters.max_velocity': '[0.16, 0.0, 1.0]',
            'controller_server.ros__parameters.progress_checker.required_movement_radius': '0.1',
            'controller_server.ros__parameters.progress_checker.movement_time_allowance': '20.0',
        })
    return LaunchDescription(_tf2_environment_actions() + [
        DeclareLaunchArgument('map', default_value=map_default),
        DeclareLaunchArgument('params_file', default_value=params_default),
        DeclareLaunchArgument('memory_enabled', default_value='true'),
        DeclareLaunchArgument('memory_path', default_value=memory_default),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('use_composition', default_value='False'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([nav2, 'launch', 'bringup_launch.py'])),
            launch_arguments={
                'map': LaunchConfiguration('map'),
                'params_file': safe_nav_params,
                'use_sim_time': 'false', 'autostart': 'true',
                'use_localization': 'true',
                'use_composition': LaunchConfiguration('use_composition'),
            }.items(),
        ),
        Node(package='smart_vision_amr', executable='astar_planner', name='astar_planner',
             parameters=[nodes, {'use_sim_time': False}], output='screen'),
        Node(package='smart_vision_amr', executable='path_follower', name='path_follower',
             parameters=[nodes, {'use_sim_time': False}], output='screen'),
        Node(package='smart_vision_amr', executable='change_memory', name='change_memory',
             parameters=[nodes, {'use_sim_time': False,
                                 'memory_enabled': ParameterValue(
                                     LaunchConfiguration('memory_enabled'), value_type=bool),
                                 'memory_path': LaunchConfiguration('memory_path')}], output='screen'),
        Node(package='rviz2', executable='rviz2', name='rviz2',
             arguments=['-d', rviz_config], parameters=[{'use_sim_time': False}],
             condition=IfCondition(LaunchConfiguration('use_rviz')), output='screen'),
    ])
