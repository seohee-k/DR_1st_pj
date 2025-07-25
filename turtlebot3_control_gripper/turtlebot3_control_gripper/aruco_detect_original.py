#!/usr/bin/env python3
import cv2
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory
import yaml
import argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage         # 📥 카메라 이미지 수신용 메시지 (압축 이미지)
from std_msgs.msg import Float32, Int32
from aruco_msgs.msg import Marker, MarkerArray     # 📤 인식된 마커 정보를 퍼블리시할 메시지 타입
from cv_bridge import CvBridge                     # ROS 이미지 <-> OpenCV 이미지 변환 도구

from visualization_msgs.msg import Marker as VisMarker                  # Rviz
from visualization_msgs.msg import MarkerArray as VisMarkerArray        # Rviz

# =====================
#  ArUco 마커 검출 및 자세 추정
# =====================
def detect_markers(image, camera_matrix, dist_coeffs, marker_size):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(image)

    detect_data = []  # [id, position, euler_angles, distance] 목록

    if ids is not None:
        # 검출된 마커 시각화
        cv2.aruco.drawDetectedMarkers(image, corners, ids)

        # 마커 위치 및 자세 추정 (rvec: 회전벡터, tvec: 위치벡터)
        rvecs, tvecs, _ = my_estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

        if rvecs is not None and tvecs is not None:
            for rvec, tvec, marker_id in zip(rvecs, tvecs, ids):
                rot_mat, _ = cv2.Rodrigues(rvec)
                yaw, pitch, roll = rotationMatrixToEulerAngles(rot_mat)  # 오일러 각도 추출
                marker_pos = np.dot(-rot_mat.T, tvec).flatten()          # 카메라 기준 좌표계에서 마커 위치
                distance = np.linalg.norm(tvec)                          #  거리 추정 (3D 벡터 크기)
                detect_data.append([marker_id, marker_pos, (yaw, pitch, roll), distance])
    return image, detect_data


# =====================
#  ArUco 마커 Pose 추정 함수
# =====================
def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    marker_points = np.array([
        [-marker_size / 2,  marker_size / 2, 0],
        [ marker_size / 2,  marker_size / 2, 0],
        [ marker_size / 2, -marker_size / 2, 0],
        [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)

    rvecs = []
    tvecs = []
    for c in corners:
        _, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
    return rvecs, tvecs, []


# =====================
#  회전 행렬 → 오일러 각 변환
# =====================
def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    return np.degrees(x), np.degrees(y), np.degrees(z)


# =====================
#  YAML 파일에서 카메라 내부 파라미터 로드
# =====================
def load_camera_parameters(yaml_file):
    package_share_directory = get_package_share_directory('aruco_detect')
    calibration_file = os.path.join(package_share_directory, 'config', yaml_file)

    with open(calibration_file, 'r') as f:
        data = yaml.safe_load(f)
        camera_matrix = np.array(data["camera_matrix"]["data"], dtype=np.float32).reshape(3, 3)
        dist_coeffs = np.array(data["distortion_coefficients"]["data"], dtype=np.float32)
    return camera_matrix, dist_coeffs


# =====================
#  ROS2 노드: ArucoMarkerDetector
# =====================
class ArucoMarkerDetector(Node):
    def __init__(self):
        super().__init__('aruco_detect')

        #  카메라 이미지 구독 (압축 이미지)
        self.subscription = self.create_subscription(
            CompressedImage,
            'image_raw/compressed',        # ← 카메라 이미지 토픽
            self.listener_callback,
            10)

        #  마커 정보 퍼블리시
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            'detected_markers',            # ← 마커 Pose/ID 퍼블리시 토픽
            10)

        # Rviz
        self.marker_viz_pub = self.create_publisher(VisMarkerArray, 'visualization_marker_array', 10)

        self.bridge = CvBridge()
        self.marker_size = 0.04  # 단위: m (마커의 실제 한 변 길이)

        #  YAML에서 카메라 파라미터 로드
        self.camera_matrix, self.dist_coeffs = load_camera_parameters('calibration_params.yaml')


    # =====================
    #  카메라 이미지 수신 시 호출되는 콜백 함수
    # =====================
    def listener_callback(self, msg):
        # 압축 이미지를 OpenCV 이미지로 디코딩
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # ArUco 마커 검출 및 거리/자세 추정
        frame, detect_data = detect_markers(frame, self.camera_matrix, self.dist_coeffs, self.marker_size)

        if len(detect_data) == 0:
            self.get_logger().debug("No markers detected")
        else:
            # 📏 가장 가까운 마커 정보 출력
            closest_marker = min(detect_data, key=lambda x: x[3])
            self.get_logger().debug(f"Closest Marker ID: {closest_marker[0]}, Distance: {closest_marker[3]:.2f}m")

            marker_array_msg = MarkerArray()
            vis_marker_array_msg = VisMarkerArray()  # RVIZ

            # MarkerArray 메시지 생성 및 퍼블리시
            marker_array_msg = MarkerArray()
            for marker in detect_data:
                marker_msg = Marker()
                marker_msg.id = int(marker[0])
                marker_msg.pose.pose.position.x = marker[1][0]
                marker_msg.pose.pose.position.y = marker[1][1]
                marker_msg.pose.pose.position.z = marker[1][2]
                marker_msg.pose.pose.orientation.x = marker[2][2]  # roll
                marker_msg.pose.pose.orientation.y = marker[2][1]  # pitch
                marker_msg.pose.pose.orientation.z = marker[2][0]  # yaw
                marker_array_msg.markers.append(marker_msg)

                # RVIZ Marker 시각화 메시지
                vis = VisMarker()
                vis.header.frame_id = "camera_link"  # 환경에 맞게 수정 가능
                vis.header.stamp = self.get_clock().now().to_msg()
                vis.ns = "aruco_marker"
                vis.id = int(marker[0])
                vis.type = VisMarker.CUBE
                vis.action = VisMarker.ADD
                vis.scale.x = self.marker_size
                vis.scale.y = self.marker_size
                vis.scale.z = 0.001
                vis.color.r = 0.0
                vis.color.g = 1.0
                vis.color.b = 0.0
                vis.color.a = 1.0
                vis.pose.position.x = marker[1][0]
                vis.pose.position.y = marker[1][1]
                vis.pose.position.z = marker[1][2]
                vis_marker_array_msg.markers.append(vis)

            self.marker_publisher.publish(marker_array_msg)
            self.marker_viz_pub.publish(vis_marker_array_msg)  # RVIZ

        # 디버깅을 위한 시각화
        cv2.imshow('Detected Markers', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoMarkerDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

