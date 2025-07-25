import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class VideoPublisher(Node):
    def __init__(self, fps=10):
        super().__init__('video_publisher')
        self.declare_and_fetch_parameters()
        self.fps = fps
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, 'video_frames', 10)
        self.setup_timer(self.fps)
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open video file: {self.video_path}')
            rclpy.shutdown()

    def declare_and_fetch_parameters(self):
        self.declare_parameter('video_path', '')
        video_path_param = self.get_parameter('video_path').get_parameter_value().string_value
        if not video_path_param:
            self.get_logger().error(
                'No video path provided. Use "--ros-args -p video_path:=<path_to_video>"'
            )
            rclpy.shutdown()
        script_dir = os.path.dirname(os.path.realpath(__file__))
        script_dir = '/'.join(script_dir.split('/')[:4])
        self.video_path = f'{script_dir}/{video_path_param}'

    def setup_timer(self, fps):
        timer_interval = 1.0 / fps
        self.timer = self.create_timer(timer_interval, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher.publish(img_msg)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

def main():
    rclpy.init()
    video_publisher = VideoPublisher(fps=30)
    rclpy.spin(video_publisher)
    video_publisher.cap.release()
    video_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
