#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import tkinter as tk
from tkinter import ttk
from PIL import Image as PILImage, ImageTk
import cv2
import numpy as np
import threading
import time

class TurtleBotControl(Node):
    def __init__(self):
        super().__init__('turtlebot_control_gui')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(Image, '/pi_camera/image_raw', self.image_callback, 10)
        self.br = CvBridge()

        self.auto_mode = True
        self.last_mode = 'center'
        self.last_corner_time = 0

        self.latest_image = None
        self.latest_mask = None

        self.linear_speed = 0.15
        self.angular_speed = 0.5

    def image_callback(self, data):
        img = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')
        self.latest_image = img.copy()
        self.process_auto_drive(img)

    def process_auto_drive(self, img):
        if not self.auto_mode:
            return  # 수동 모드일 때는 자동 주행 안함

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([22, 150, 150])
        upper_yellow = np.array([32, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        self.latest_mask = mask.copy()

        h, w = mask.shape
        roi = mask[int(h * 0.7):, :]
        left_roi = roi[:, :int(w * 0.2)]
        right_roi = roi[:, int(w * 0.8):]

        M_left = cv2.moments(left_roi)
        M_right = cv2.moments(right_roi)

        twist = Twist()
        now = time.time()

        # 양쪽 선이 보일 때: 중앙 주행
        if M_left['m00'] > 0 and M_right['m00'] > 0:
            cx_left = int(M_left['m10'] / M_left['m00'])
            cx_right = int(M_right['m10'] / M_right['m00']) + int(w * 0.8)
            cx = (cx_left + cx_right) // 2
            err = cx - w // 2
            twist.linear.x = self.linear_speed
            twist.angular.z = -float(err) / 120.0
            self.last_mode = 'center'

        # 왼쪽만 사라짐 → 오른쪽 선만 인식됨
        elif M_right['m00'] > 0:
            if self.last_mode != 'right':
                self.last_corner_time = now
            self.last_mode = 'right'
            twist.linear.x = 0.2
            twist.angular.z = 0.2

        # 오른쪽만 사라짐 → 왼쪽 선만 인식됨
        elif M_left['m00'] > 0:
            if self.last_mode != 'left':
                self.last_corner_time = now
            self.last_mode = 'left'
            twist.linear.x = 0.2
            twist.angular.z = -0.2
    
        # 선이 모두 안 보이는 경우: 멈추거나 회전 유지
        else:
            twist.linear.x = 0.05
            twist.angular.z = 0.2

        self.publisher_.publish(twist)

    def send_manual_cmd(self, lin, ang):
        if self.auto_mode:
            return
        twist = Twist()
        twist.linear.x = lin
        twist.angular.z = ang
        self.publisher_.publish(twist)

    def stop(self):
        self.send_manual_cmd(0.0, 0.0)

    def toggle_mode(self):
        self.auto_mode = not self.auto_mode
        self.get_logger().info(f"[모드 전환] 자동 모드: {'ON' if self.auto_mode else 'OFF'}")
        if self.auto_mode:
            self.stop()

def run_gui(node: TurtleBotControl):
    win = tk.Tk()
    win.title("TurtleBot3 Controller")

    speed_var = tk.DoubleVar(value=node.linear_speed)

    def update_speed(val):
        node.linear_speed = float(val)

    # 속도 조절
    tk.Label(win, text="속도 조절").pack()
    tk.Scale(win, from_=0.05, to=0.5, resolution=0.01, orient=tk.HORIZONTAL,
             variable=speed_var, command=update_speed).pack()

    # 조작 버튼
    btn_frame = tk.Frame(win)
    btn_frame.pack()

    tk.Button(btn_frame, text="↑ 전진", width=10,
              command=lambda: node.send_manual_cmd(node.linear_speed, 0.0)).grid(row=0, column=1)
    tk.Button(btn_frame, text="← 좌회전", width=10,
              command=lambda: node.send_manual_cmd(0.0, node.angular_speed)).grid(row=1, column=0)
    tk.Button(btn_frame, text="정지", width=10, command=node.stop).grid(row=1, column=1)
    tk.Button(btn_frame, text="→ 우회전", width=10,
              command=lambda: node.send_manual_cmd(0.0, -node.angular_speed)).grid(row=1, column=2)
    tk.Button(btn_frame, text="↓ 후진", width=10,
              command=lambda: node.send_manual_cmd(-node.linear_speed, 0.0)).grid(row=2, column=1)

    # 모드 전환
    def toggle():
        node.toggle_mode()
        btn_mode.config(text="자동 → 수동" if not node.auto_mode else "수동 → 자동")

    btn_mode = tk.Button(win, text="수동 → 자동", command=toggle)
    btn_mode.pack(pady=5)

    # 카메라 뷰
    label = tk.Label(win)
    label.pack()

    def update_view():
        if node.latest_mask is not None:
            mask_bgr = cv2.cvtColor(node.latest_mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(mask_bgr, f"Mode: {node.last_mode}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            img_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
            im = PILImage.fromarray(img_rgb)
            imgtk = ImageTk.PhotoImage(image=im)
            label.imgtk = imgtk
            label.configure(image=imgtk)
        label.after(50, update_view)

    update_view()
    win.mainloop()

def main(args=None):
    rclpy.init(args=args)
    node = TurtleBotControl()
    gui_thread = threading.Thread(target=run_gui, args=(node,), daemon=True)
    gui_thread.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
