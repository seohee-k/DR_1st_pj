from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='turtlebot3_move',
            executable='line_follower',
            name='line_follower',
            output='screen',
        )
    ])
