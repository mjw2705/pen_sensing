import cv2
import time
import numpy as np
import mediapipe as mp
import math

from utils import *


THRESHOLD = 35
# PATH = './video/left_ex1.mp4'
PATH = './video/ex3.mp4'
FILM = 'right'

'''기하학적 변환 행렬 계산 사용'''

class Main(object):
    def __init__(self):
        super(Main, self).__init__()

        self.click_chk = input('클릭?: ')

        self.hand = mp.solutions.hands.Hands(static_image_mode=False,
                                            max_num_hands=1,
                                            min_detection_confidence=0.5,
                                            min_tracking_confidence=0.5)

        self.holistic = mp.solutions.holistic.Holistic(static_image_mode=False, 
                                                        min_detection_confidence=0.6,
                                                        min_tracking_confidence=0.5)
        self.ptime = 0
        self.frame_shape = (852, 480)
        self.screen_w, self.screen_h = 1677, 966
        self.pen, self.gesture = False, False
        self.click_pts = []
        self.hand_lms = []
        self.pose_lms = []
        self.mode = None
        
        self.cap = cv2.VideoCapture(PATH)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_shape[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_shape[1])
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # out = cv2.VideoWriter('step1.avi', fourcc, 30, frame_shape)

    def click_monitor_corner(self):
        if self.click_chk == 'y':
            success, first_image = self.cap.read()
            self.click_pts = click_monitor(first_image)
            print(self.click_pts)
            cv2.destroyWindow('first_image')

        else:
            if FILM == 'left':
                self.click_pts = np.array([[231, 82], [228, 322], [535, 128], [535, 311]])
            else:
                self.click_pts = np.array([[122, 108], [115, 331], [509, 85], [509, 347]])

    def calc_geometric_transform(self):
        # 기하학적 변환 행렬 계산
        m_monitor = np.array([[0, 0, self.screen_w, self.screen_w], [0, self.screen_h, 0, self.screen_h], [0, 0, 0, 0], [0, 0, 0, 0]])
        c_monitor = np.array([self.click_pts[:, 0], self.click_pts[:, 1], self.click_pts[:, 0] * self.click_pts[:, 1], [1, 1, 1, 1]])
        c_monitor_inv = np.linalg.inv(c_monitor)
        T = m_monitor.dot(c_monitor_inv)

        return T

    def detect_pen(self, THRESHOLD):
        x1, y1 = self.hand_lms[4][0], self.hand_lms[4][1]
        x2, y2 = self.hand_lms[8][0], self.hand_lms[8][1]
        abs_x1, abs_y1, abs_x2, abs_y2 = int(x1 * self.frame_shape[0]), int(y1 * self.frame_shape[1]), \
            int(x2 * self.frame_shape[0]), int(y2 * self.frame_shape[1])

        length = math.hypot(abs_x2 - abs_x1, abs_y2 - abs_y1)

        if length < THRESHOLD:
            self.pen = True
            self.gesture = False
            self.mode = 'pen'
        else:
            self.pen = False
            self.gesture = True
            self.mode = 'gesture'

    def process(self):
        self.click_monitor_corner()

        T = self.calc_geometric_transform()

        while self.cap.isOpened():
            success, image = self.cap.read()
            h, w, _ = image.shape

            if not success:
                print("Ignoring empty camera frame.")
                break
                # continue
                
            point1, point2, point3, point4 = draw_roi(self.click_pts.tolist(), image)
            screen_roi = image[point1[1]:point4[1], point1[0]:point4[0]].copy()
            roi_h, roi_w, _ = screen_roi.shape
            cv2.imshow('screen', screen_roi)
            
            # 손 detect
            self.hand_lms, self.pose_lms = holistics(self.holistic, screen_roi)
            
            if self.hand_lms is None:
                pass
            else:
                # 손 roi
                start, end = get_hand_rect(self.hand_lms, (roi_w, roi_h), offset=point1, box_ratio=4)
                sx, sy = start
                ex, ey = end
                cv2.rectangle(image, (sx, sy), (ex, ey), (0, 0, 255), 2)

                self.detect_pen(THRESHOLD)

                if self.pen is True:
                    # 펜촉 위치
                    pen_x, pen_y = self.hand_lms[0][0:2]
                    pen_abs_x, pen_abs_y = int(pen_x * roi_w + point1[0]), int(pen_y * roi_h + point1[1])

                    P_x, P_y, _, _ = T.dot([pen_abs_x, pen_abs_y, 1, 1])
                    cv2.circle(image, (pen_abs_x, pen_abs_y), 5, (0, 255, 0), -1, cv2.LINE_AA)
                    if FILM == 'left':
                        cv2.putText(image, f"x:{int(P_x)}, y:{int(P_y)}", (10, 460), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)
                    else:
                        cv2.putText(image, f"x:{int(P_x)}, y:{int(P_y)}", (620, 460), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)

            ctime = time.time()
            fps = 1/(ctime-self.ptime)
            self.ptime = ctime
            cv2.putText(image, f"fps: {str(int(fps))}", (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)

            if FILM == 'left':
                cv2.putText(image, f"mode: {self.mode}", (10, 420), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)
            else:
                cv2.putText(image, f"mode: {self.mode}", (620, 420), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)

            cv2.imshow('image', image)
            # out.write(image)

            if cv2.waitKey(1) == 27:
                break

        # out.release()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    mainSensing = Main()
    mainSensing.process()