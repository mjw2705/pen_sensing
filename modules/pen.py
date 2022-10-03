import cv2
import numpy as np
import math
from modules.util import Status

class Pen:
    def __init__(self):
        self.mode = None
        self.hand_lms = None
        self.pen_color = [255, 255, 255]

    '''소리가 들어오면 변경해야됨
        임시 함수'''
    def detect_pen(self, hand_lms):
        self.hand_lms = hand_lms
        x1, y1 = self.hand_lms[4][0], self.hand_lms[4][1]
        x2, y2 = self.hand_lms[8][0], self.hand_lms[8][1]
        abs_x2, abs_y2 = int(x2 * Status.width), int(y2 * Status.height)
        abs_x1, abs_y1 = int(x1 * Status.width), int(y1 * Status.height)

        length = math.hypot(abs_x2 - abs_x1, abs_y2 - abs_y1)
        # length = math.hypot(x2 - x1, y2 - y1)

        if length < 33:
            self.mode = 'pen'
            return True
        else:
            self.mode = 'gesture'
            return False

    def _calc_distance(self, a, b):
        x1, y1 = a
        x2, y2 = b

        len_x = abs(x2 - x1)
        len_y = abs(y2 - y1)

        return math.sqrt(math.pow(len_x, 2) * math.pow(len_y, 2))
    
    def _goodfeature(self, image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(img_gray, 2, 0.1, 20, blockSize=9, useHarrisDetector=True, k=0.05)
        return corners
    
    def locate_pen(self, image, hand_img, point_s):
        img_hsv = cv2.cvtColor(hand_img, cv2.COLOR_BGR2HSV)
        # color = cv2.inRange(img_hsv, color_value - threshold, color_value + threshold)
        color = cv2.inRange(img_hsv, (0,0,140),(172,111,225))
        # color = cv2.inRange(img_hsv, (20, 100, 100),(30, 255, 255))
        range_image = cv2.bitwise_and(hand_img, hand_img, mask=color)
        cv2.imshow('range_image', range_image)
        cv2.imshow('color', color)

        ## 코너 찾는 다른 방법
        # corners = self._goodfeature(range_image)
        # if corners is not None:
        #     idx = np.argmin(corners, axis=0)[0][-1]
        #     x, y = corners[idx].ravel()
        #     P_x, P_y = int(x + point_s[0]), int(y + point_s[1])
        #     cv2.circle(image, (P_x, P_y), 5, (0, 255, 0), -1, cv2.LINE_AA)

        corners = self._find_corner(range_image)
        if corners is not None:
            idx = np.argmin(corners, axis=0)[-1]
            x, y = corners[idx].ravel()
            P_x, P_y = int(x + point_s[0]), int(y + point_s[1])
            cv2.circle(image, (P_x, P_y), 5, (0, 255, 0), -1, cv2.LINE_AA)
        return P_x, P_y

    def locate_pen2(self, image, hand_img, point_s):
        corners = self._goodfeature(hand_img)

        if corners is not None:
            for i in corners:
                x, y = i.ravel()
                P_x, P_y = int(x + point_s[0]), int(y + point_s[1])
                cv2.circle(image, (P_x, P_y), 5, (0, 255, 0), -1, cv2.LINE_AA)

        return P_x, P_y


    def _find_corner(self, hand_img):
        img_gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
        img_gray = np.float32(img_gray)
        dst = cv2.cornerHarris(img_gray, 2, 3, 0.1)
        dst = cv2.dilate(dst, None)

        ret, dst = cv2.threshold(dst,0.05*dst.max(),255,0) 
        dst = np.uint8(dst) 
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100 ,0.001) 
        corners = cv2.cornerSubPix(img_gray, np.float32(centroids),(5,5),(-1,-1),criteria)
        return corners
        # 손 중점에서 가장 먼 pentip 계산
        m_dis = 0
        P_x, P_y = 0, 0
        for corner in corners:
            dis = max(m_dis, self._calc_distance(corner, self.hand_lms[0][:2]))
            if dis == m_dis:
                P_x, P_y = corner
        return P_x, P_y




#=============================================================================================#
    '''image에서 펜촉위치 찾고 기하학적 변환할 때 사용'''
    def geometric_locate_pen(self, image, hand_img, point_s, click_pts):
        corners = self._find_pen_tip(hand_img)
        for x, y in corners:
            abs_x, abs_y = int(x + point_s[0]), int(y + point_s[1])
            
            T = self._calc_geometric_transform(click_pts)
            P_x, P_y, _, _ = T.dot([abs_x, abs_y, 1, 1])
            cv2.circle(image, (abs_x, abs_y), 5, (0, 255, 0), -1, cv2.LINE_AA)

        return P_x, P_y

    def _find_pen_tip(self, hand_img):
        img_gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
        img_gray = np.float32(img_gray)
        dst = cv2.cornerHarris(img_gray, 2, 3, 0.1)
        dst = cv2.dilate(dst, None)

        ret, dst = cv2.threshold(dst,0.05*dst.max(),255,0) 
        dst = np.uint8(dst) 
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100 ,0.001) 
        corners = cv2.cornerSubPix(img_gray, np.float32(centroids),(5,5),(-1,-1),criteria) 
        return corners

    def _calc_geometric_transform(self, click_pts):
        m_monitor = np.array([
            [0, 0, Status.monitor_w, Status.monitor_w], 
            [0, Status.monitor_h, 0, Status.monitor_h],
            [0, 0, 0, 0], 
            [0, 0, 0, 0]
        ])
        c_monitor = np.array([
            click_pts[:, 0],
            click_pts[:, 1],
            click_pts[:, 0] * click_pts[:, 1],
            [1, 1, 1, 1]
        ])

        c_monitor_inv = np.linalg.inv(c_monitor)
        T = m_monitor.dot(c_monitor_inv)
        
        return T