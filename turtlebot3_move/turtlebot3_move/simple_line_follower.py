#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class SimpleCornerLineFollower(Node):
    def __init__(self):
        super().__init__('simple_corner_line_follower')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(Image, '/pi_camera/image_raw', self.listener_callback, 10)
        self.br = CvBridge()
        self.last_mode = 'center'
        self.last_corner_time = 0

    def listener_callback(self, data):
        img = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([22, 150, 150])
        upper_yellow = np.array([32, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        h, w = mask.shape
        roi = mask[int(h*0.6):, :]
        left_roi = roi[:, :int(w*0.2)]
        right_roi = roi[:, int(w*0.8):]

        M_left = cv2.moments(left_roi)
        M_right = cv2.moments(right_roi)

        twist = Twist()
        now = time.time()

        # 양쪽 선이 보일 때: 중앙 주행
        if M_left['m00'] > 0 and M_right['m00'] > 0:
            cx_left = int(M_left['m10'] / M_left['m00'])
            cx_right = int(M_right['m10'] / M_right['m00']) + int(w * 0.8)
            cx = (cx_left + cx_right) // 2
            err = cx - w // 2
            twist.linear.x = 0.15
            twist.angular.z = -float(err) / 120.0
            self.last_mode = 'center'

        # 왼쪽만 사라짐 → 오른쪽 선만 인식됨
        elif M_right['m00'] > 0:
            if self.last_mode != 'right':
                self.last_corner_time = now
            self.last_mode = 'right'
            twist.linear.x = 0.10
            twist.angular.z = 0.3  # 왼쪽으로 회전

        # 오른쪽만 사라짐 → 왼쪽 선만 인식됨
        elif M_left['m00'] > 0:
            if self.last_mode != 'left':
                self.last_corner_time = now
            self.last_mode = 'left'
            twist.linear.x = 0.10
            twist.angular.z = -0.3  # 오른쪽으로 회전

        # 선이 모두 안 보이는 경우: 멈추거나 회전 유지
        else:
            twist.linear.x = 0.05
            twist.angular.z = 0.2  # 기본 회전

        self.publisher_.publish(twist)

        # 시각화
        debug_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(debug_img, f"Mode: {self.last_mode}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.imshow("Line Follower View", debug_img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleCornerLineFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()