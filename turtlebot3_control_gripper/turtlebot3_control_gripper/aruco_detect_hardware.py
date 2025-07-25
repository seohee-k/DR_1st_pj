#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import math

from cv2 import aruco
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from aruco_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

class ArucoHardwareDetector(Node):
    def __init__(self):
        super().__init__('aruco_hardware_detector')

        # [★] 카메라 내부 파라미터 임의로 직접 세팅 (drgo wc720, 640x480 기준)
        self.camera_matrix = np.array([
            [570.0,   0.0, 320.0],
            [  0.0, 570.0, 240.0],
            [  0.0,   0.0,   1.0]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((5,), dtype=np.float32)  # 왜곡 없음 가정

        # ArUco 설정
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.marker_size = 0.04  # (m, 4cm)

        self.bridge = CvBridge()

        # 카메라 이미지 구독 (실제 하드웨어 카메라 토픽)
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',   # 하드웨어 환경에 맞춰 사용
            self.image_callback,
            10)

        # 디버그 이미지 퍼블리시 (RViz2 등에서 확인)
        self.debug_image_pub = self.create_publisher(Image, '/aruco/debug_image', 10)

        # MarkerArray 퍼블리시
        self.publisher = self.create_publisher(MarkerArray, '/detected_markers', 10)

        # PoseStamped 퍼블리시 (가장 가까운 마커)
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco_target_pose', 10)

    # [★] camera_info_callback은 사용하지 않으므로 제거

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        corners, ids, _ = self.aruco_detector.detectMarkers(frame)
        marker_array = MarkerArray()

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

            for i in range(len(ids)):
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]

                # XYZ 축 시각화
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.02)

                # 회전 → 오일러 → 쿼터니언
                rot_mat, _ = cv2.Rodrigues(rvec)
                roll, pitch, yaw = self.rotationMatrixToEulerAngles(rot_mat)
                quat = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()

                marker = Marker()
                marker.id = int(ids[i][0])
                marker.pose.pose.position.x = float(tvec[0])
                marker.pose.pose.position.y = float(tvec[1])
                marker.pose.pose.position.z = float(tvec[2])
                marker.pose.pose.orientation.x = float(quat[0])
                marker.pose.pose.orientation.y = float(quat[1])
                marker.pose.pose.orientation.z = float(quat[2])
                marker.pose.pose.orientation.w = float(quat[3])
                marker_array.markers.append(marker)

            # 가장 가까운 마커 PoseStamped로 퍼블리시
            closest_idx = np.argmin([np.linalg.norm(t) for t in tvecs])
            rvec = rvecs[closest_idx][0]
            tvec = tvecs[closest_idx][0]
            rot_mat, _ = cv2.Rodrigues(rvec)
            roll, pitch, yaw = self.rotationMatrixToEulerAngles(rot_mat)
            quat = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()

            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "base_link"  # 필요에 따라 camera_link로 변경

            pose_msg.pose.position.x = float(tvec[0])
            pose_msg.pose.position.y = float(tvec[1])
            pose_msg.pose.position.z = float(tvec[2])
            pose_msg.pose.orientation.x = float(quat[0])
            pose_msg.pose.orientation.y = float(quat[1])
            pose_msg.pose.orientation.z = float(quat[2])
            pose_msg.pose.orientation.w = float(quat[3])

            self.pose_pub.publish(pose_msg)

        self.publisher.publish(marker_array)

        # 시각화/디버그
        cv2.imshow("ArUco View", frame)
        cv2.waitKey(1)

        debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.debug_image_pub.publish(debug_msg)

    def rotationMatrixToEulerAngles(self, R):
        sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0
        return roll, pitch, yaw

def main(args=None):
    rclpy.init(args=args)
    node = ArucoHardwareDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()