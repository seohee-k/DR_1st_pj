#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from yolov8_msgs.msg import Yolov8Inference


class Yolov8Subscriber(Node):
    def __init__(self):
        super().__init__('yolov8_subscriber')
        self.subscription = self.create_subscription(
            Yolov8Inference,
            '/yolov8/detections',
            self.detection_callback,
            10)
        self.get_logger().info("YOLOv8 서브스크라이버 노드 시작됨")

    def detection_callback(self, msg):
        for det in msg.detections:
            self.get_logger().info(
                f"[감지됨] 클래스: {det.class_name}, 신뢰도: {det.confidence:.2f}, 위치: ({det.xmin}, {det.ymin}) ~ ({det.xmax}, {det.ymax})")

            if det.class_name.lower() == "stop_sign":
                width = det.xmax - det.xmin
                height = det.ymax - det.ymin
                area = width * height
                if area > 5:  # 너무 작지 않은 경우에만 반응
                    self.get_logger().warn("🚨 STOP SIGN 감지됨! 정지 신호를 수행해야 함")


def main(args=None):
    rclpy.init(args=args)
    node = Yolov8Subscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
