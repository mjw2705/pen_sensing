import cv2
import numpy as np
import math
from modules.util import Status

class Pen:
    def __init__(self):
        self.pen = False
        self.gesture = False
        self.mode = None
        self.hand_lms = None
        self.pen_color = [255, 255, 255]

    def detect_pen(self, hand_lms):
        self.hand_lms = hand_lms
        x1, y1 = self.hand_lms[4][0], self.hand_lms[4][1]
        x2, y2 = self.hand_lms[8][0], self.hand_lms[8][1]
        abs_x2, abs_y2 = int(x2 * Status.width), int(y2 * Status.height)
        abs_x1, abs_y1 = int(x1 * Status.width), int(y1 * Status.height)

        length = math.hypot(abs_x2 - abs_x1, abs_y2 - abs_y1)
        # length = math.hypot(x2 - x1, y2 - y1)

        if length < 33:
            self.pen = True
            self.gesture = False
            self.mode = 'pen'
            return True
        else:
            self.pen = False
            self.gesture = True
            self.mode = 'gesture'
            return False

    def get_hand_hull(self, image, hand_img, offset):
        h, w = hand_img.shape[:2]
        max_value = np.max(self.hand_lms, axis=0)
        min_value = np.min(self.hand_lms, axis=0)
        sx = max(int(min_value[0] * w), 0)
        ex = min(int(max_value[0] * w), w)
        sy = max(int(min_value[1] * h), 0)
        ey = min(int(max_value[1] * h), h)
        cv2.rectangle(hand_img, (sx, sy), (ex, ey), (255, 0, 0), 2)
        return sx, sy, ex, ey

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
    
    def _find_pen_corner(self, hand_img):
        img_gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
        img_gray = np.float32(img_gray)
        dst = cv2.cornerHarris(img_gray, 2, 3, 0.1)
        dst = cv2.dilate(dst, None)

        ret, dst = cv2.threshold(dst,0.05*dst.max(),255,0) 
        dst = np.uint8(dst) 
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100 ,0.001) 
        corners = cv2.cornerSubPix(img_gray, np.float32(centroids),(5,5),(-1,-1),criteria)
        m_dis = 0
        pen_tip = 0, 0
        for corner in corners:
            dis = max(m_dis, self._calc_distance(corner, self.hand_lms[0][:2]))
            if dis == m_dis:
                pen_tip = corner
        return pen_tip

    def _find_pen_corner2(self, hand_img):
        img_gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
        img_gray = np.float32(img_gray)
        dst = cv2.cornerHarris(img_gray, 2, 3, 0.1)
        dst = cv2.dilate(dst, None)

        ret, dst = cv2.threshold(dst,0.05*dst.max(),255,0) 
        dst = np.uint8(dst) 
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100 ,0.001) 
        corners = cv2.cornerSubPix(img_gray, np.float32(centroids),(5,5),(-1,-1),criteria)
        res = np.concatenate((centroids, corners))
        dis = 0
        pen_tip = 0, 0
        for corner in res:
            dis = max(dis, self._calc_distance(corner, self.hand_lms[0][:2]))
        pen_tip = corner
        return pen_tip
    
    def _goodfeature(self, hand_img):
        h, w = hand_img.shape[:2]
        img_gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(img_gray, 2, 0.1, 20, blockSize=9, useHarrisDetector=True, k=0.05)
        pen_tip = 0, 0
        dis = 0
        cx, cy = self.hand_lms[0][0] * w, self.hand_lms[0][1] * h
        return corners
        # pix = np.array(hand_img)

        # if corners is None:
        #     pen_tip = None
        # else:
        #     # 손과 가장 먼 거리 점 찾기
        #     for i in corners:
        #         x, y = i.ravel()
        #         c_dis = dis
        #         dis = max(dis, self._calc_distance((x, y), (cx, cy)))
        #         if dis != c_dis:
        #             pen_tip = (x, y)

            # for i in corners:
            #     x, y = i.ravel()
            #     if list(pix[int(y)][int(x)]) == self.pen_color:
            #         pen_tip = (x, y)

        return pen_tip

    def _calc_distance(self, a, b):
        x1, y1 = a
        x2, y2 = b

        len_x = abs(x2 - x1)
        len_y = abs(y2 - y1)
        return math.sqrt(math.pow(len_x, 2) * math.pow(len_y, 2))

    def _locate_pen1(self, image, hand_img, point_s, click_pts):
        corners = self._find_pen_tip(hand_img)
        for x, y in corners:
            abs_x, abs_y = int(x + point_s[0]), int(y + point_s[1])
            
            T = self._calc_geometric_transform(click_pts)
            P_x, P_y, _, _ = T.dot([abs_x, abs_y, 1, 1])
            cv2.circle(image, (abs_x, abs_y), 5, (0, 255, 0), -1, cv2.LINE_AA)

        return P_x, P_y

    def locate_pen2(self, image, hand_img, point_s):
        # pen_tip = self._goodfeature(hand_img)
        # if pen_tip is not None:
        #     x, y = pen_tip
        #     abs_x, abs_y = int(x + point_s[0]), int(y + point_s[1])
        #     cv2.circle(image, (abs_x, abs_y), 5, (0, 255, 0), -1, cv2.LINE_AA)
        # else:
        #     abs_x, abs_y = 0, 0
        
        corners = self._goodfeature(hand_img)
        if corners is not None:
            for i in corners:
                x, y = i.ravel()
                abs_x, abs_y = int(x + point_s[0]), int(y + point_s[1])
                cv2.circle(image, (abs_x, abs_y), 5, (0, 255, 0), -1, cv2.LINE_AA)
        
        # x, y = self._find_pen_corner2(hand_img)
        # abs_x, abs_y = int(x + point_s[0]), int(y + point_s[1])
        # cv2.circle(image, (abs_x, abs_y), 5, (0, 255, 0), -1, cv2.LINE_AA)
        return abs_x, abs_y 


    def locate_pen(self, frame_shape, point1, image, click_pts):
        roi_h, roi_w = frame_shape
        pen_x, pen_y = self.pen_lms[0][0:2]
        pen_abs_x, pen_abs_y = int(pen_x * roi_w + point1[0]), int(pen_y * roi_h + point1[1])

        T = self._calc_geometric_transform(click_pts)
        P_x, P_y, _, _ = T.dot([pen_abs_x, pen_abs_y, 1, 1])
        cv2.circle(image, (pen_abs_x, pen_abs_y), 5, (0, 255, 0), -1, cv2.LINE_AA)

        return P_x, P_y

    def detect_pen_color(self, image, hand_img, point_s):
        img_hsv = cv2.cvtColor(hand_img, cv2.COLOR_BGR2HSV)
        img_mask = cv2.inRange(img_hsv, (245, 245, 245), (255, 255, 255))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_DILATE, kernel, iterations=3)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(img_mask)

        abs_x, abs_y = int(centroids[0][0] + point_s[0]), int(centroids[0][1] + point_s[1])
        cv2.circle(image, (abs_x, abs_y), 5, (0, 255, 0), -1, cv2.LINE_AA)
        return abs_x, abs_y

    def detect_pen_color2(self, image, hand_img, point_s):
        img_hsv = cv2.cvtColor(hand_img, cv2.COLOR_BGR2HSV)
        color = cv2.inRange(img_hsv, (0, 95, 165), (10, 255, 255))
        thresh = cv2.dilate(color, kernel=np.ones((5,5),np.uint8), iterations = 1)
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow('tt', img_hsv)

        max_area = 0
        best_cnt = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                best_cnt = cnt
        print(best_cnt)
        # finding centroids of best_cnt and draw a circle there
        M = cv2.moments(best_cnt)
        cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        # cv2.circle(img_hsv,(cx,cy),8,[0, 0, 0],5)
        # cv2.circle(img_hsv,(cx,cy),8,[0, 255, 0],4)
        return cx, cy

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