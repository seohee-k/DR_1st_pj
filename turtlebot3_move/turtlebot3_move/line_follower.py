import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
from slide_window import SlideWindow  # 너가 만든 클래스 불러옴

class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')
        self.bridge = CvBridge()
        self.sw = SlideWindow()

        self.image_sub = self.create_subscription(
            Image,
            '/pi_camera/image_raw',  # 카메라 토픽에 맞게 수정
            self.image_callback,
            10
        )

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        result, left_x, right_x, _ = self.sw.slide(binary)

        twist = Twist()

        if result:
            center = int((left_x + right_x) / 2)
            error = (frame.shape[1] // 2) - center
            twist.linear.x = 0.15
            twist.angular.z = float(error) * 0.005
        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
