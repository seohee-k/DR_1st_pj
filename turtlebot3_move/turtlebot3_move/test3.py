#!/usr/bin/env python3

# ROS2 및 OpenCV, Tkinter 등 사용
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from yolov8_msgs.msg import Yolov8Inference  # YOLOv8 검출 결과 메시지
from cv_bridge import CvBridge
import cv2
import numpy as np
import tkinter as tk
from PIL import Image as PILImage, ImageTk
import threading
import time

# -----------------------------
# 🧠 핵심 노드 클래스 정의
# -----------------------------
class TurtleBotYoloGui(Node):
    def __init__(self):
        super().__init__('turtlebot3_gui_yolo')

        # /cmd_vel 토픽으로 속도 명령을 퍼블리시할 퍼블리셔 생성
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        # 카메라 이미지와 YOLO 검출 결과를 구독
        self.image_sub = self.create_subscription(Image, '/pi_camera/image_raw', self.image_callback, 10)
        self.yolo_sub = self.create_subscription(Yolov8Inference, '/yolov8/detections', self.yolo_callback, 10)

        self.br = CvBridge()  # ROS 이미지 ↔ OpenCV 변환용 브리지
        self.latest_image = None  # 최근 카메라 이미지 저장
        self.latest_mask = None   # 최근 라인 마스크 이미지 저장
        self.latest_yolo_image = None  # 최근 YOLO 결과 이미지 저장

        # 상태 변수들
        self.auto_mode = True  # 자동 주행 모드 여부
        self.last_mode = 'center'  # 마지막 라인트레이싱 방향
        self.linear_speed = 0.15   # 기본 직진 속도
        self.angular_speed = 0.5   # 기본 회전 속도
        self.pause_requested = False  # 정지 요청 여부
        self.last_pause_time = 0      # 마지막 정지 시간
        self.status_text = "주행 중"  # 상태 표시 텍스트
        self.crossing_thread = None   # 정지 후 전진 스레드

    # --------------------------------
    # 📸 카메라 이미지 콜백 (라인 추적)
    # 카메라 이미지가 들어오면 최신 이미지를 저장하고,
    # 자동 모드일 때 라인트레이싱 로직을 실행
    def image_callback(self, data):
        img = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')
        self.latest_image = img.copy()
        if self.auto_mode:
            self.process_line_following(img)

    # --------------------------------
    # 🧠 YOLO 객체 인식 콜백
    # YOLO 검출 결과가 들어오면, 각 객체에 대해 바운딩 박스와 클래스명을 이미지에 표시
    # 'stop sign'이 일정 크기 이상 감지되면 정지 요청 및 일정 시간 후 전진
    def yolo_callback(self, msg):
        if self.latest_image is None:
            return

        img = self.latest_image.copy()

        for det in msg.yolov8_inference:
            class_name = det.class_name.lower()
            self.get_logger().info(f"[YOLO] Detected: {class_name}")

            # 바운딩 박스 좌표 추출
            x1, y1 = int(det.left), int(det.top)
            x2, y2 = int(det.right), int(det.bottom)

            # 클래스명 해시로 색상 결정
            color = (hash(class_name) % 256, (hash(class_name)*2) % 256, (hash(class_name)*3) % 256)

            # 바운딩 박스와 클래스명 이미지에 표시
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, class_name, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # STOP SIGN 감지 시 정지 및 crossing 스레드 실행
            if class_name in ["stop", "stop_sign", "stop sign"]:
                area = (x2 - x1) * (y2 - y1)
                # 일정 크기 이상 & 최근 정지 이후 5초 이상 경과 시
                if area > 26000 and (time.time() - self.last_pause_time) > 5:
                    self.get_logger().info("[YOLO] STOP SIGN 감지됨 → 정지 요청")

                    if not self.pause_requested:
                        self.pause_requested = True
                        self.last_pause_time = time.time()
                        self.status_text = "STOP SIGN 감지 → 정지 중..."

                        # 즉시 정지
                        stop_twist = Twist()
                        stop_twist.linear.x = 0.0
                        stop_twist.angular.z = 0.0
                        self.publisher_.publish(stop_twist)

                        # 스레드 시작
                        if self.crossing_thread is None or not self.crossing_thread.is_alive():
                            self.crossing_thread = threading.Thread(target=self._handle_crossing, daemon=True)
                            self.crossing_thread.start()


        # YOLO 결과 이미지 갱신
        self.latest_yolo_image = img

    # --------------------------------
    # 🛑 정지 후 일정 시간 전진
    # STOP SIGN 감지 후 3초 정지, 이후 10초간 직진
    def _handle_crossing(self):
        time.sleep(3)  # 3초 정지
        self.status_text = "주행 재개"

        forward_twist = Twist()
        forward_twist.linear.x = self.linear_speed

        # 10초간 직진 명령 퍼블리시
        start_time = time.time()
        while time.time() - start_time < 10.0:
            self.publisher_.publish(forward_twist)
            time.sleep(0.1)

        self.pause_requested = False

    # --------------------------------
    # 🟡 라인트레이싱 로직
    # 노란색 라인을 추적하여 Twist 명령 생성 및 퍼블리시
    def process_line_following(self, img):
        if self.pause_requested:
            return  # 정지 요청이 있을 경우 라인트레이싱 동작 중단

        # BGR 이미지를 HSV 색공간으로 변환 (노란색 검출에 유리)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([22, 150, 150])   # 노란색 하한값 (HSV)
        upper_yellow = np.array([32, 255, 255])   # 노란색 상한값 (HSV)
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)  # 노란색 영역만 마스킹
        self.latest_mask = mask.copy()  # GUI에서 표시할 마스크 이미지 저장

        h, w = mask.shape
        # 카메라의 하단 40%만 사용하여 ROI(관심영역) 설정
        # 코너 주행 시나 바닥의 다른 선과 혼동하지 않도록, 하단만 추적
        roi = mask[int(h * 0.6):, :]
        # ROI 중 왼쪽 20%, 오른쪽 20%만 각각 별도로 추적
        # 카메라의 양쪽 20퍼센트씩, 하단의 40퍼센트만 사용하도록 하여 
        # 자율주행 시, 코너 주행 시 카메라에 보이는 다른 실선과 혼동하지 않도록 설정하였습니다.
        left_roi = roi[:, :int(w * 0.2)]   # 왼쪽 20%
        right_roi = roi[:, int(w * 0.8):]  # 오른쪽 20%

        # 왼쪽/오른쪽 ROI 각각의 무게중심 계산
        M_left = cv2.moments(left_roi)
        M_right = cv2.moments(right_roi)

        twist = Twist()  # ROS2 Twist 메시지 생성 (속도/회전 명령)

    # 아래는 각 상황별 주행 로직이 이어집니다.

        # 왼쪽/오른쪽 모두 라인 감지 시 중앙 유지
        if M_left['m00'] > 0 and M_right['m00'] > 0:
            cx_left = int(M_left['m10'] / M_left['m00'])
            cx_right = int(M_right['m10'] / M_right['m00']) + int(w * 0.8)
            cx = (cx_left + cx_right) // 2
            err = cx - w // 2
            twist.linear.x = self.linear_speed
            twist.angular.z = -float(err) / 120.0
            self.last_mode = 'center'
        # 오른쪽만 감지 시 우회전
        elif M_right['m00'] > 0:
            twist.linear.x = 0.1
            twist.angular.z = 0.3
            self.last_mode = 'right'
        # 왼쪽만 감지 시 좌회전
        elif M_left['m00'] > 0:
            twist.linear.x = 0.1
            twist.angular.z = -0.3
            self.last_mode = 'left'
        # 둘 다 감지 안되면 느리게 직진
        else:
            twist.linear.x = 0.05
            twist.angular.z = 0.2
            self.last_mode = 'lost'

        self.publisher_.publish(twist)

    # --------------------------------
    # 🕹 수동 제어
    # 수동 모드에서만 Twist 명령 퍼블리시
    def send_manual_cmd(self, lin, ang):
        if self.auto_mode:
            return
        twist = Twist()
        twist.linear.x = lin
        twist.angular.z = ang
        self.publisher_.publish(twist)

    # 정지 명령
    def stop(self):
        self.send_manual_cmd(0.0, 0.0)

    # 자동/수동 모드 전환
    def toggle_mode(self):
        self.auto_mode = not self.auto_mode
        self.status_text = "자동 주행 ON" if self.auto_mode else "수동 조작 중"
        self.get_logger().info(f"[모드 전환] 자동 모드: {'ON' if self.auto_mode else 'OFF'}")
        if self.auto_mode:
            self.stop()

# --------------------------------
# 🖥️ GUI 구현 (Tkinter)
# --------------------------------
def run_gui(node: TurtleBotYoloGui):
    win = tk.Tk()  # Tkinter 메인 윈도우 생성
    win.title("TurtleBot3 GUI + YOLOv8")  # 윈도우 제목 설정

    speed_var = tk.DoubleVar(value=node.linear_speed)  # 속도 조절용 변수(슬라이더와 연동)

    def update_speed(val):
        node.linear_speed = float(val)  # 슬라이더 값이 바뀔 때 로봇 속도에 반영

    tk.Label(win, text="속도 조절").pack()  # 속도 조절 라벨
    tk.Scale(win, from_=0.05, to=0.5, resolution=0.01, orient=tk.HORIZONTAL,
             variable=speed_var, command=update_speed).pack()  # 속도 조절 슬라이더

    btn_frame = tk.Frame(win)  # 방향 버튼을 담을 프레임
    btn_frame.pack()

    # 방향 버튼들 (수동 조작 시 사용)
    tk.Button(btn_frame, text="↑ 전진", width=10,
              command=lambda: node.send_manual_cmd(node.linear_speed, 0.0)).grid(row=0, column=1)
    tk.Button(btn_frame, text="← 좌회전", width=10,
              command=lambda: node.send_manual_cmd(0.0, node.angular_speed)).grid(row=1, column=0)
    tk.Button(btn_frame, text="정지", width=10, command=node.stop).grid(row=1, column=1)
    tk.Button(btn_frame, text="→ 우회전", width=10,
              command=lambda: node.send_manual_cmd(0.0, -node.angular_speed)).grid(row=1, column=2)
    tk.Button(btn_frame, text="↓ 후진", width=10,
              command=lambda: node.send_manual_cmd(-node.linear_speed, 0.0)).grid(row=2, column=1)

    def toggle():
        # 자동/수동 모드 전환 및 버튼 텍스트 갱신
        node.toggle_mode()
        btn_mode.config(text="자동 → 수동" if not node.auto_mode else "수동 → 자동")

    btn_mode = tk.Button(win, text="수동 → 자동", command=toggle)  # 모드 전환 버튼
    btn_mode.pack(pady=5)

    status_label = tk.Label(win, text="주행 상태", font=("Arial", 14))  # 상태 표시 라벨
    status_label.pack(pady=5)

    # 카메라 이미지 출력 프레임 생성 (라인/YOLO 결과를 각각 표시)
    frame = tk.Frame(win)
    frame.pack()
    label_line = tk.Label(frame)      # 라인트레이싱(노란색 마스크) 이미지를 표시할 라벨
    label_line.pack(side=tk.LEFT)
    label_yolo = tk.Label(frame)      # YOLO 객체 인식 결과 이미지를 표시할 라벨
    label_yolo.pack(side=tk.RIGHT)

    def update_view():
        # 상태 텍스트(예: "주행 중", "정지 중" 등)를 GUI에 표시
        status_label.config(text=node.status_text)

        # 라인트레이싱 마스크 이미지가 있을 때 화면에 표시
        if node.latest_mask is not None:
            # 흑백 마스크를 컬러로 변환
            mask_bgr = cv2.cvtColor(node.latest_mask, cv2.COLOR_GRAY2BGR)
            # 현재 라인트레이싱 모드(중앙/좌/우/로스트)를 이미지에 텍스트로 표시
            cv2.putText(mask_bgr, f"Mode: {node.last_mode}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            # OpenCV 이미지를 RGB로 변환 후 PIL 이미지로 변환
            img_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
            im = PILImage.fromarray(img_rgb)
            imgtk = ImageTk.PhotoImage(image=im)
            # Tkinter 라벨에 이미지 업데이트
            label_line.imgtk = imgtk
            label_line.configure(image=imgtk)

        # YOLO 객체 인식 결과 이미지가 있을 때 화면에 표시
        if node.latest_yolo_image is not None:
            # OpenCV 이미지를 RGB로 변환 후 PIL 이미지로 변환
            yolo_rgb = cv2.cvtColor(node.latest_yolo_image, cv2.COLOR_BGR2RGB)
            im2 = PILImage.fromarray(yolo_rgb)
            imgtk2 = ImageTk.PhotoImage(image=im2)
            # Tkinter 라벨에 이미지 업데이트
            label_yolo.imgtk = imgtk2
            label_yolo.configure(image=imgtk2)

        # 50ms마다 update_view 함수를 반복 호출하여 실시간 갱신
        label_line.after(50, update_view)

    update_view()    # 최초 호출로 주기적 갱신 시작
    win.mainloop()   # Tkinter GUI 이벤트 루프 시작

# --------------------------------
# 🚀 메인 함수
# --------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = TurtleBotYoloGui()
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
