#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np

class YoloV8Detector(Node):
    def __init__(self):
        super().__init__('yolov8_detector')
        self.get_logger().info("YOLOv8 detector node initialized")

        # 카메라 이미지 구독
        self.subscription = self.create_subscription(
            Image,
            '/pi_camera/image_raw',
            self.image_callback,
            10)

        self.br = CvBridge()

        # YOLOv8 모델 로드 (경로 확인 필수)
        self.model = YOLO('/home/djqsp2/yolov8n.pt')  # yolov8n.pt or yolov8s.pt, etc.

        # 클래스 이름 (COCO)
        self.class_names = self.model.names

    def image_callback(self, msg):
        # ROS Image → OpenCV
        frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # YOLOv8 추론
        results = self.model(frame)[0]

        # 결과 시각화
        annotated_frame = frame.copy()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{self.class_names[cls_id]} {conf:.2f}"

            # 박스 좌표
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # GUI 창에 표시
        cv2.imshow("YOLOv8 Detection", annotated_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YoloV8Detector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
