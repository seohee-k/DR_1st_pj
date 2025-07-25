#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from yolov8_msgs.msg import Yolov8Inference
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

# 주행을 시작하는 클래스
class StartDrive:
    def __init__(self, node):
        self.publisher = node.create_publisher(Twist, '/cmd_vel', 10)

    def publish(self, linear=0.0, angular=0.0):
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.publisher.publish(twist)

# 라인 트레이싱 로직
class LineFollowerLogic:
    def __init__(self):
        self.last_mode = 'center'
        self.last_corner_time = 0

    def process(self, img, w, h):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([22, 150, 150])
        upper_yellow = np.array([32, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        roi = mask[int(h*0.6):, :]
        left_roi = roi[:, :int(w*0.2)]
        right_roi = roi[:, int(w*0.8):]

        M_left = cv2.moments(left_roi)
        M_right = cv2.moments(right_roi)

        now = time.time()
        twist = Twist()

        if M_left['m00'] > 0 and M_right['m00'] > 0:
            cx_left = int(M_left['m10'] / M_left['m00'])
            cx_right = int(M_right['m10'] / M_right['m00']) + int(w * 0.8)
            cx = (cx_left + cx_right) // 2
            err = cx - w // 2
            twist.linear.x = 0.15
            twist.angular.z = -float(err) / 120.0
            self.last_mode = 'center'

        elif M_right['m00'] > 0:
            if self.last_mode != 'right':
                self.last_corner_time = now
            self.last_mode = 'right'
            twist.linear.x = 0.10
            twist.angular.z = 0.3

        elif M_left['m00'] > 0:
            if self.last_mode != 'left':
                self.last_corner_time = now
            self.last_mode = 'left'
            twist.linear.x = 0.10
            twist.angular.z = -0.3

        else:
            twist.linear.x = 0.05
            twist.angular.z = 0.2

        return twist, mask, self.last_mode

# 정지 및 재출발
class PauseDrive:
    def __init__(self, node):
        self.node = node
        self.publisher = node.create_publisher(Twist, '/cmd_vel', 10)

    def execute(self, duration=3.0, resume_linear=0.15, resume_angular=0.0):
        stop_twist = Twist()
        stop_twist.linear.x = 0.0
        stop_twist.angular.z = 0.0
        self.publisher.publish(stop_twist)
        self.node.get_logger().info(f"[PauseDrive] STOP SIGN → 정지 중... ({duration}초)")
        time.sleep(duration)

        resume_twist = Twist()
        resume_twist.linear.x = resume_linear
        resume_twist.angular.z = resume_angular
        self.publisher.publish(resume_twist)
        self.node.get_logger().info("[PauseDrive] 주행 재개")

# 전체 노드
class SimpleCornerLineFollower(Node):
    def __init__(self):
        super().__init__('simple_corner_line_follower')
        self.br = CvBridge()
        self.drive = StartDrive(self)
        self.pause = PauseDrive(self)
        self.logic = LineFollowerLogic()

        self.subscription = self.create_subscription(Image, '/pi_camera/image_raw', self.listener_callback, 10)
        self.yolo_sub = self.create_subscription(Yolov8Inference, '/yolov8/detections', self.yolo_callback, 10)

        self.pause_requested = False
        self.last_pause_time = 0

    def yolo_callback(self, msg):
        for det in msg.detections:
            if det.class_name.lower() == "stop_sign":
                width = det.xmax - det.xmin
                height = det.ymax - det.ymin
                area = width * height
                if area > 30:  # stop 사인이 충분히 클 때만 정지
                    now = time.time()
                    if now - self.last_pause_time > 5:  # 최소 5초 간격 두기
                        self.pause_requested = True
                        self.last_pause_time = now

    def listener_callback(self, data):
        if self.pause_requested:
            self.pause.execute(3.0)
            self.pause_requested = False
            return

        img = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')
        h, w, _ = img.shape

        twist, mask, mode = self.logic.process(img, w, h)
        self.drive.publish(twist.linear.x, twist.angular.z)

        debug_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(debug_img, f"Mode: {mode}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
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
