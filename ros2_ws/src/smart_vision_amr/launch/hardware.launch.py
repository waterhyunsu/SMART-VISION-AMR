"""PC-side navigation. Start turtlebot3_bringup robot.launch.py on the Pi separately."""

import os
from pathlib import Path
from xml.etree import ElementTree

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, GroupAction, IncludeLaunchDescription,
    OpaqueFunction, SetEnvironmentVariable,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node, SetParameter, SetRemap
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _validate_localization_mode(context):
    mode = LaunchConfiguration('localization_mode').perform(context)
    if mode != 'baseline':
        raise ValueError('This navigation package supports baseline localization only')
    return []


def _tf2_environment_actions():
    """Preserve the previously verified TF overlay on affected Humble hosts."""
    manifest = Path('/opt/ros/humble/share/tf2/package.xml')
    if not manifest.exists():
        return []
    version = ElementTree.parse(manifest).getroot().findtext('version', default='0.0.0')
    numbers = tuple(int(part) for part in version.split('.')[:3])
    if not (0, 25, 21) <= numbers <= (0, 25, 24):
        return []
    prefix = Path.home() / '.local/share/smart_vision_amr/tf2-0.25.20/root/opt/ros/humble'
    if not (prefix / 'lib/libtf2.so').exists() or not (prefix / 'lib/libtf2_ros.so').exists():
        raise RuntimeError(f'tf2 {version} requires the verified tf2 0.25.20 overlay at {prefix}')
    libraries = str(prefix / 'lib')
    python_paths = [str(prefix / 'lib/python3.10/site-packages'),
                    str(prefix / 'local/lib/python3.10/dist-packages')]
    return [
        SetEnvironmentVariable('LD_LIBRARY_PATH', os.pathsep.join(
            part for part in (libraries, os.environ.get('LD_LIBRARY_PATH')) if part)),
        SetEnvironmentVariable('PYTHONPATH', os.pathsep.join(
            part for part in python_paths + [os.environ.get('PYTHONPATH')] if part)),
    ]


def generate_launch_description():
    own = FindPackageShare('smart_vision_amr')
    nav2 = FindPackageShare('nav2_bringup')
    map_default = PathJoinSubstitution([own, 'maps', 'classroom_20261027_cleaned.yaml'])
    params_default = PathJoinSubstitution([own, 'config', 'curve_tuning_waffle_pi.yaml'])
    nodes_default = PathJoinSubstitution([own, 'config', 'classroom_hardware_nodes.yaml'])
    nodes = LaunchConfiguration('nodes_file')
    rviz_config = PathJoinSubstitution([own, 'rviz', 'minimal.rviz'])
    memory_default = '~/.ros/smart_vision_amr/classroom_memory.json'
    return LaunchDescription(_tf2_environment_actions() + [
        DeclareLaunchArgument('map', default_value=map_default),
        DeclareLaunchArgument('params_file', default_value=params_default),
        DeclareLaunchArgument('nodes_file', default_value=nodes_default),
        DeclareLaunchArgument('memory_enabled', default_value='true'),
        DeclareLaunchArgument('memory_path', default_value=memory_default),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('localization_mode', default_value='baseline',
                              description='baseline (stock TurtleBot3 odometry)'),
        DeclareLaunchArgument('use_composition', default_value='false',
                              description='Run Nav2 in a shared container; false uses separate processes'),
        DeclareLaunchArgument('lifecycle_bond_timeout', default_value='15.0',
                              description='Nav2 lifecycle bond wait, seconds, for separate-process bringup'),
        DeclareLaunchArgument('use_motion_mux', default_value='true',
                              description='Disable only when the web service owns the velocity mux'),
        OpaqueFunction(function=_validate_localization_mode),
        GroupAction([
            # The classroom PC can be slow to schedule map_server during
            # concurrent Nav2 startup. Keep bond failure detection enabled,
            # but allow a longer connection window for separate processes.
            SetParameter(name='bond_timeout',
                         value=ParameterValue(LaunchConfiguration('lifecycle_bond_timeout'),
                                              value_type=float),
                         condition=UnlessCondition(LaunchConfiguration('use_composition'))),
            # Preserve controller -> cmd_vel_nav -> smoother wiring. Remap
            # only the final outputs, including recovery behavior commands.
            SetRemap(src='velocity_smoother:cmd_vel_smoothed',
                     dst='/smart_vision_amr/navigation_cmd_vel'),
            SetRemap(src='behavior_server:cmd_vel',
                     dst='/smart_vision_amr/navigation_cmd_vel'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(PathJoinSubstitution([nav2, 'launch', 'bringup_launch.py'])),
                launch_arguments={
                    'map': LaunchConfiguration('map'),
                    'params_file': LaunchConfiguration('params_file'),
                    'use_sim_time': 'false', 'autostart': 'true',
                    'use_localization': 'true',
                    # Humble's child launch evaluates ``not <value>`` as a
                    # Python expression, so pass True/False even for lower-case CLI values.
                    'use_composition': PythonExpression([
                        "'", LaunchConfiguration('use_composition'), "'.lower() in ('true', '1')"]),
                }.items(),
            ),
        ]),
        Node(package='smart_vision_amr', executable='motion_mux', name='motion_mux',
             parameters=[nodes, {'use_sim_time': False}], output='screen',
             condition=IfCondition(LaunchConfiguration('use_motion_mux'))),
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
