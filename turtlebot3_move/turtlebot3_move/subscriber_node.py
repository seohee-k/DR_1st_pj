import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
import cv2
import numpy as np
from turtlebot3_move import slide_window
from turtlebot3_move import camera_processing

class VideoSubscriber(Node):
    def __init__(self):
        super().__init__('video_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/pi_camera/image_raw',
            self.listener_callback,
            10
        )
        self.image_publisher = self.create_publisher(Image, 'processed_frames', 10)
        self.marker_publisher = self.create_publisher(Marker, 'lane_info_marker', 10)
        self.bridge = CvBridge()

        self.camera_processor = camera_processing.CameraProcessing()
        self.slide_window_processor = slide_window.SlideWindow()

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
	
        
	
        detected, left, right, processed = self.lane_detect(frame)
        
        processed_msg = self.bridge.cv2_to_imgmsg(processed, encoding='bgr8')
        self.image_publisher.publish(processed_msg)

        info_text = f'Left position: {left}, Right position: {right}'

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.z = 2.0
        marker.scale.z = 0.5
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.text = info_text
        self.marker_publisher.publish(marker)

    def lane_detect(self, frame):
        frame, filtered = self.camera_processor.process_image(frame)

        if frame is not None:
            slide_frame = frame[frame.shape[0] - 200:frame.shape[0] - 150, :]
            detected, left, right, tmp_frame = self.slide_window_processor.slide(slide_frame)
            processed_frame = self.slide_window_processor.lane_visualization(frame,left,right)
            self.processed_frame = processed_frame
            return detected, left, right, self.processed_frame
        return False, None, None, frame

def main(args=None):
    rclpy.init(args=args)
    video_subscriber = VideoSubscriber()
    rclpy.spin(video_subscriber)
    cv2.destroyAllWindows()
    video_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

