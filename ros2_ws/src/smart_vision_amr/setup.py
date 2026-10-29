from glob import glob
from setuptools import setup

package_name = 'smart_vision_amr'
setup(
    name=package_name, version='0.1.0', packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/maps', glob('maps/*')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='SMART-VISION-AMR', maintainer_email='maintainer@example.invalid',
    description='Real TurtleBot3 Waffle Pi A* navigation', license='MIT',
    entry_points={'console_scripts': [
        'astar_planner = smart_vision_amr.astar_planner:main',
        'path_follower = smart_vision_amr.path_follower:main',
        'change_memory = smart_vision_amr.change_memory:main',
        'motion_mux = smart_vision_amr.motion_mux:main',
    ]},
)
