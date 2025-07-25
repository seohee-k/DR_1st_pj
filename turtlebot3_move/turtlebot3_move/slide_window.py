import numpy as np
import cv2

class SlideWindow:
    def __init__(self):
        self.center_x = 320  # 초기 중심 (이미지 중심 기준)

    def slide(self, img):
        height, width = img.shape
        c_img = np.dstack((img, img, img))  # 시각화용

        # 슬라이딩 윈도우 설정
        window_height = 20
        window_width = 30
        minpix = 40
        y = height - window_height - 10  # 아래에서 위로 한 줄만 검사

        # 중심선 표시 (흰색 점선)
        cv2.line(c_img, (width // 2, 0), (width // 2, height), (255, 255, 255), 1)

        # 중심 기준 윈도우 좌표
        win_x_low = int(self.center_x - window_width // 2)
        win_x_high = int(self.center_x + window_width // 2)
        win_y_low = y
        win_y_high = y + window_height

        # 윈도우 시각화
        cv2.rectangle(c_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (255, 0, 0), 2)

        # 유효 픽셀 추출
        nonzero = img[win_y_low:win_y_high, win_x_low:win_x_high].nonzero()
        nonzerox = nonzero[1] + win_x_low
        nonzeroy = nonzero[0] + win_y_low

        # 픽셀이 일정 수 이상이면 중심 업데이트
        if len(nonzerox) > minpix:
            new_center = int(np.mean(nonzerox))
            self.center_x = new_center
            # 인식된 점 시각화
            for x, y in zip(nonzerox, nonzeroy):
                cv2.circle(c_img, (x, y), 1, (255, 0, 255), -1)
            return True, self.center_x, c_img
        else:
            # 인식 실패 → 이전 위치 유지
            return False, self.center_x, c_img
