from setuptools import Extension, setup
from glob import glob

package_name = 'smart_vision_amr'
setup(
    name=package_name, version='0.2.0', packages=[package_name],
    ext_modules=[Extension(package_name + '._geometry_native',
        [package_name + '/_geometry_native.cpp'], language='c++', optional=True,
        extra_compile_args=['-O3', '-fno-fast-math', '-ffp-contract=off'])],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy', 'opencv-python-headless'], zip_safe=True,
    maintainer='waterhyunsu',
    maintainer_email='286576468+waterhyunsu@users.noreply.github.com',
    description='A* navigation, local detours and execution-aware recovery extracted from SMART-VISION-AMR',
    license='MIT',
    entry_points={'console_scripts': [
        'astar_planner = smart_vision_amr.astar_planner:main',
        'path_follower = smart_vision_amr.path_follower:main',
        'change_memory = smart_vision_amr.change_memory:main',
        'motion_mux = smart_vision_amr.motion_mux:main',
        'command_stamper = smart_vision_amr.command_stamper:main',
        'localization_health = smart_vision_amr.localization_health:main',
    ]},
)
