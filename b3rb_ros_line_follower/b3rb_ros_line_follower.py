# Copyright 2024 NXP

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Joy

import math
import time
import cv2
import numpy as np

from synapse_msgs.msg import EdgeVectors
from synapse_msgs.msg import TrafficStatus
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int64

QOS_PROFILE_DEFAULT = 10

PI = math.pi

LEFT_TURN = +1.0
RIGHT_TURN = -1.0

TURN_MIN = 0.0
TURN_MAX = 1.0
SPEED_MIN = 0.0
SPEED_MAX = 1.0
SPEED_25_PERCENT = SPEED_MAX / 4
SPEED_50_PERCENT = SPEED_25_PERCENT * 2
SPEED_75_PERCENT = SPEED_25_PERCENT * 3

THRESHOLD_OBSTACLE_VERTICAL = 0.40
THRESHOLD_OBSTACLE_HORIZONTAL = 0.07
COLLISION_THRESHOLD = 0.30
COLLISION_THRESHOLD_HORIZONTAL = 0.0


class LineFollower(Node):
    """ Initializes line follower node with the required publishers and subscriptions.

        Returns:
            None
    """
    def __init__(self):
        super().__init__('line_follower')

        # Subscription for edge vectors.
        self.subscription_vectors = self.create_subscription(
            EdgeVectors,
            '/edge_vectors',
            self.edge_vectors_callback,
            QOS_PROFILE_DEFAULT)

        # Publisher for joy (for moving the rover in manual mode).
        self.publisher_joy = self.create_publisher(
            Joy,
            '/cerebri/in/joy',
            QOS_PROFILE_DEFAULT)

        # Subscription for traffic status.
        self.subscription_traffic = self.create_subscription(
            TrafficStatus,
            '/traffic_status',
            self.traffic_status_callback,
            QOS_PROFILE_DEFAULT)

        # Subscription for LIDAR data.
        self.subscription_lidar = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            QOS_PROFILE_DEFAULT)
        
        self.publisher_obstacle = self.create_publisher(
            Int64,
            '/obstacle',
            QOS_PROFILE_DEFAULT)

        # Initialize status variables.
        self.traffic_status = TrafficStatus()
        self.obstacle_detected = False
        self.free_side = None
        self.left_obstacle_detected = False
        self.right_obstacle_detected = False
        self.collision_detected = False
        self.counter = 0
        self.prev_turn = 0.0

    """ Moves the rover in manual mode based on given speed and turn values. """
    def rover_move_manual_mode(self, speed, turn):
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        msg.axes = [0.0, speed, 0.0, turn]
        self.publisher_joy.publish(msg)

    """ Callback for edge vector data to achieve line following. """
    def edge_vectors_callback(self, message):
        speed = SPEED_50_PERCENT*1.5
        turn = TURN_MIN

        vectors = message
        half_width = vectors.image_width / 2
        obstacles = Int64()

        if self.counter == 0:
            self.collision_detected = False
        # Slow down for stop signs
        if self.traffic_status.stop_sign:
            speed = SPEED_MIN
            self.get_logger().info("Stop sign detected")
            self.rover_move_manual_mode(speed, 0.0)
            time.sleep(2.0)
            return
        if self.traffic_status.left_turn:
            speed = SPEED_50_PERCENT
            turn = LEFT_TURN/2
            self.rover_move_manual_mode(speed, turn)
            self.get_logger().info("Left sign detected")
            return
        if self.traffic_status.right_turn:
            speed = SPEED_50_PERCENT
            turn = RIGHT_TURN/2
            self.rover_move_manual_mode(speed, turn)
            self.get_logger().info("Right sign detected")
            return
        if self.traffic_status.straight and self.obstacle_detected:
            speed = SPEED_50_PERCENT
            turn = 0.0
            self.rover_move_manual_mode(speed, turn)
            self.get_logger().info("Straight sign detected")
            return

        # Update obstacle status for publishing
        if self.left_obstacle_detected and self.right_obstacle_detected:
            obstacles.data = 0
        elif self.obstacle_detected:
            obstacles.data = 1
        else:
            obstacles.data = 0

        self.publisher_obstacle.publish(obstacles)

        if self.traffic_status.stop_sign is False or self.traffic_status.left_turn is False or self.traffic_status.right_turn is False or self.traffic_status.straight is False:
            # Obstacle avoidance logic
            if (self.obstacle_detected or self.left_obstacle_detected or self.right_obstacle_detected):
                self.get_logger().info("Obstacle detected")
                if not self.collision_detected:
                    speed = SPEED_50_PERCENT*0.6
                    # Steering based on free side
                    if self.left_obstacle_detected:
                        turn = RIGHT_TURN * 0.0
                    elif self.right_obstacle_detected:
                        turn = LEFT_TURN * 0.0
                    else:
                        turn = LEFT_TURN if self.free_side == "Left" else RIGHT_TURN * 0.75
                    self.rover_move_manual_mode(speed, turn)
                elif self.collision_detected and self.counter:
                    # Collision detected, reverse
                    speed = -SPEED_25_PERCENT
                    turn = -1*self.prev_turn
                    self.rover_move_manual_mode(speed, TURN_MIN)
                    self.counter = self.counter - 1
                return

            # Line following logic
            if vectors.vector_count == 0:
                speed = SPEED_50_PERCENT*1.5
            elif vectors.vector_count == 1:
                deviation = vectors.vector_1[1].x - vectors.vector_1[0].x
                turn = deviation / vectors.image_width
                print(deviation)
                if abs(turn) > 0.2:
                    turn = np.sign(turn)*TURN_MAX
                    speed = SPEED_50_PERCENT
                else:
                    turn = deviation / vectors.image_width
                    speed = SPEED_50_PERCENT*1.5
            elif vectors.vector_count == 2:
                middle_x_left = (vectors.vector_1[0].x + vectors.vector_1[1].x) / 2
                middle_x_right = (vectors.vector_2[0].x + vectors.vector_2[1].x) / 2
                middle_x = (middle_x_left + middle_x_right) / 2
                deviation = half_width - middle_x
                turn = deviation / vectors.image_width

        
        self.prev_turn = turn
        self.rover_move_manual_mode(speed, turn)

    """ Callback for traffic status. """
    def traffic_status_callback(self, message):
        self.traffic_status = message

    """ LIDAR callback for obstacle and ramp detection. """
    def lidar_callback(self, message):
        length = len(message.ranges)
        ranges = message.ranges[int(3*length/4): ]+ message.ranges[:int(length/4)]
        length = len(ranges)
        front_ranges = ranges[int(0.375*length): int(0.625*length)]
        side_ranges_right = ranges[:int(length/4)]
        side_ranges_left = ranges[int(3*length/4):]

        front_ranges_right = front_ranges[:len(front_ranges) // 2]
        front_ranges_left = front_ranges[len(front_ranges) // 2:]

        # Check for front obstacles
        if any(r < THRESHOLD_OBSTACLE_VERTICAL for r in front_ranges):
            self.obstacle_detected = True
            self.free_side = "Left" if min(front_ranges_left) > min(front_ranges_right) else "Right"
            self.get_logger().info(f"Obstacle detected, free side: {self.free_side}")
        else:
            self.obstacle_detected = False

        # Check for side obstacles
        self.left_obstacle_detected = any(r < THRESHOLD_OBSTACLE_HORIZONTAL for r in side_ranges_left)
        self.right_obstacle_detected = any(r < THRESHOLD_OBSTACLE_HORIZONTAL for r in side_ranges_right)
        
        # Handle possible collision detection
        if any(r < COLLISION_THRESHOLD for r in front_ranges):
            self.obstacle_detected = True
            self.collision_detected = True
            self.counter = 20

def main(args=None):
    rclpy.init(args=args)
    line_follower = LineFollower()
    rclpy.spin(line_follower)
    line_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
