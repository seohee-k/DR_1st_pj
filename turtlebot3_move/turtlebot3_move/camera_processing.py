import numpy as np
import cv2

class CameraProcessing:
    def __init__(self):
        self.GaussianBlur = 3 # 커질수록 더 부드러워짐
        self.LightRemove = 15 # 조명 제거 강도(작아질 수록 많이 없어짐)
        self.MedianBlur = 3
        self.bin_threshold = 10
        self.kernels = {
            'vertical': np.array([[-1,  0,  1],
                                  [-2,  0,  2],
                                  [-1,  0,  1]]),
            'diagonal_1': np.array([[ 0,  1,  2],
                                    [-1,  0,  1],
                                    [-2, -1,  0]]),
            'diagonal_2': np.array([[ 2,  1,  0],
                                    [ 1,  0, -1],
                                    [ 0, -1, -2]])
        }

    def process_image(self, img):
        if img is None:
            return None

        img = self.remove_lighting(img)
        _, img = cv2.threshold(img, self.bin_threshold, 255, cv2.THRESH_BINARY)

        img = cv2.medianBlur(img, self.MedianBlur)

        img = self.warp(img)
        filtered, img = self.choose_filtered_img(img)
        return img, filtered

    def choose_filtered_img(self, img):
        max_edge_strength = -1
        best_filtered_img = None
        best_kernel_name = None

        for kernel_name, kernel in self.kernels.items():
            filtered_img = cv2.filter2D(img, -1, kernel)
            edge_strength = np.sum(np.abs(filtered_img))

            if edge_strength > max_edge_strength:
                max_edge_strength = edge_strength
                best_filtered_img = filtered_img
                best_kernel_name = kernel_name

        return best_kernel_name, best_filtered_img

    def remove_lighting(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.LightRemove, self.LightRemove), 0)
        diff_img = cv2.subtract(gray, blurred)
        normalized_img = cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX)
        return normalized_img

    def warp(self, img):
        h, w = img.shape[:2]
        src = np.float32([
            [170, 300], #top-left
            [40, 380],  #bottom-left
            [500, 300], #top-right
            [630, 380], #bottom-right
        ])
        dst = np.float32([
            [140, 0],
            [140, 480],
            [500, 0],
            [500, 480],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        warped_img = cv2.warpPerspective(img, M, (w, h))
        return warped_img
