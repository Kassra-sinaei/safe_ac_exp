from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('vicon_receiver'), 
            '/launch/client.launch.py'
        ])
        ),
        Node(
            package='unitree_legged_real',
            executable='ros2_udp',
            name='go1_driver',
            output='log',
            arguments=['highlevel']
        ),
        Node(
            package='safe_ac_exp',
            executable='communication_node',
            name='comm_node',
            output='screen'
        ),
        Node(
            package='safe_ac_exp',
            executable='omni_control.py',
            name='controller_node',
            output='screen'
        ),
    ])