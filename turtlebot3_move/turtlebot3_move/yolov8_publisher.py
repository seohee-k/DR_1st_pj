#!/usr/bin/env python3

# ROS2, 이미지 처리, YOLO 관련 모듈 임포트
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from yolov8_msgs.msg import Yolov8Inference, InferenceResult
from std_msgs.msg import Header
from cv_bridge import CvBridge
import torch
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------
# 🧠 YOLOv8 인식 + 퍼블리시 노드 정의
# ---------------------------------------------
class Yolov8Publisher(Node):
    def __init__(self):
        super().__init__('yolov8_publisher')

        # ROS 이미지 ↔ OpenCV 이미지 변환용 브릿지
        self.bridge = CvBridge()

        # ✅ YOLOv8 모델 로드 (.pt 파일은 사전에 학습된 모델 경로)
        model_path = "/home/djqsp2/yolobot/src/yolobot_recognition/scripts/yolov8n.pt"
        self.model = YOLO(model_path)

        # 📸 카메라 이미지 구독
        self.subscription = self.create_subscription(
            Image, '/pi_camera/image_raw', self.image_callback, 10)

        # 📤 객체 인식 결과 퍼블리셔 (사용자 정의 메시지)
        self.publisher_ = self.create_publisher(Yolov8Inference, '/yolov8/detections', 10)

        # 클래스 이름 저장 (예: ['person', 'car', 'stop sign' ...])
        self.class_names = self.model.names

    # ---------------------------------------------
    # 📸 이미지 콜백 - YOLO 인식 수행
    # ---------------------------------------------
    def image_callback(self, msg):
        # ROS 이미지 → OpenCV 이미지로 변환
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # YOLO 모델 추론 (한 프레임에 대한 결과)
        results = self.model(cv_image, verbose=False)[0]

        # Yolov8Inference 메시지 생성
        yolov8_msg = Yolov8Inference()
        yolov8_msg.header = Header()
        yolov8_msg.header.stamp = self.get_clock().now().to_msg()

        # 🔄 결과 반복: 박스마다 InferenceResult로 포장
        for box in results.boxes:
            r = InferenceResult()
            r.class_name = self.class_names[int(box.cls)]  # 클래스명 문자열
            r.left = int(box.xyxy[0][0].item())
            r.top = int(box.xyxy[0][1].item())
            r.right = int(box.xyxy[0][2].item())
            r.bottom = int(box.xyxy[0][3].item())
            yolov8_msg.yolov8_inference.append(r)

        # ✅ 결과 퍼블리시
        self.publisher_.publish(yolov8_msg)
        self.get_logger().info(f"Published {len(yolov8_msg.yolov8_inference)} detections.")

# ---------------------------------------------
# 🚀 메인 함수: 노드 실행
# ---------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = Yolov8Publisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
