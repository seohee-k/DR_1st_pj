#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import math

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from aruco_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R


class ArucoSimulDetector(Node):
    def __init__(self):
        super().__init__('aruco_simul_detector')

        # 초기 카메라 파라미터 비워두기
        self.camera_matrix = None
        self.dist_coeffs = None

        # ArUco 설정
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.marker_size = 0.04  # 단위: meter (4cm)

        self.bridge = CvBridge()

        # 카메라 이미지 구독
        self.subscription = self.create_subscription(
            Image,
            '/pi_camera/image_raw',
            self.image_callback,
            10)

        # 카메라 파라미터 구독
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/pi_camera/camera_info',
            self.camera_info_callback,
            10)

        # 디버그 이미지 퍼블리시 (RViz 확인용)
        self.debug_image_pub = self.create_publisher(Image, '/aruco/debug_image', 10)

        # 마커 배열 퍼블리시 (MarkerArray)
        self.publisher = self.create_publisher(MarkerArray, '/detected_markers', 10)

        # 가장 가까운 마커 PoseStamped 퍼블리시
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco_target_pose', 10)

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.get_logger().info("📷 카메라 파라미터를 수신하여 초기화합니다.")

            self.camera_matrix = np.array(msg.k, dtype=np.float32).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d, dtype=np.float32)

            self.get_logger().info(f"Camera Matrix (K):\n{self.camera_matrix}")
            self.get_logger().info(f"Distortion Coeffs (D): {self.dist_coeffs}")

    def image_callback(self, msg: Image):
        if self.camera_matrix is None:
            self.get_logger().warn("⏳ 카메라 파라미터 수신 전. 이미지 처리 대기 중...")
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        corners, ids, _ = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.aruco_params)

        marker_array = MarkerArray()

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

            for i in range(len(ids)):
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]

                # XYZ 축 시각화
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs,
                                  rvec, tvec, 0.02)

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
            pose_msg.header.frame_id = "base_link"  # 카메라 프레임과 일치시켜야 함

            pose_msg.pose.position.x = float(tvec[0])
            pose_msg.pose.position.y = float(tvec[1])
            pose_msg.pose.position.z = float(tvec[2])
            pose_msg.pose.orientation.x = float(quat[0])
            pose_msg.pose.orientation.y = float(quat[1])
            pose_msg.pose.orientation.z = float(quat[2])
            pose_msg.pose.orientation.w = float(quat[3])

            self.pose_pub.publish(pose_msg)

        self.publisher.publish(marker_array)

        cv2.imshow("Gazebo ArUco View", frame)
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
    node = ArucoSimulDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()
