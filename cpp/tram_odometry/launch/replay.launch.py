"""Launch the fallback odometry node.

    ros2 launch tram_odometry replay.launch.py use_clock:=true

Use scripts/run_bag.sh to start the node and replay a bag in one go; the node
itself never spawns ros2 bag play, so the jury can drive it manually too.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('tram_odometry')
    default_params = os.path.join(pkg, 'config', 'params.yaml')

    args = [
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument(
            'use_clock', default_value='true',
            description='use /clock; set to false when the bag is played without --clock'),
    ]

    node = Node(
        package='tram_odometry',
        executable='odometry_node',
        name='tram_odometry',
        output='screen',
        emulate_tty=True,
        parameters=[
            LaunchConfiguration('params_file'),
            {'use_sim_time': ParameterValue(LaunchConfiguration('use_clock'), value_type=bool)},
        ],
    )

    return LaunchDescription(args + [node])
