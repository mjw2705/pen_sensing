import cv2
import numpy as np
from modules.util import *


pts = np.zeros((4, 2), dtype=np.float32)
pts_cnt = 0

def click_event(event, x, y, flags, param):
    global pts_cnt, pts

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(param, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)
        pts[pts_cnt] = [x, y]
        pts_cnt += 1

class ClickManager:
    def __init__(self):
        global pts_cnt, pts
        self.click_pts = None
    
    def click_monitor(self, cap):
        success, first_image = cap.read()
        image = cv2.resize(first_image, (Status.width, Status.height))

        if Status.click_state:
            while pts_cnt < 4:
                cv2.imshow('first_image', image)
                cv2.setMouseCallback('first_image', click_event, image)
                cv2.waitKey(1)

                if cv2.waitKey(1) == 27:
                    cv2.destroyWindow('first_image')
                    sys.exit("No click")
                    # return None
                    
            if pts_cnt == 4:
                sm = pts.sum(axis=1)
                diff = np.diff(pts, axis=1)

                top_left = pts[np.argmin(sm)]
                top_right = pts[np.argmin(diff)]
                bottom_left = pts[np.argmax(diff)]
                bottom_right = pts[np.argmax(sm)]

                # 좌상, 좌하, 우상, 우하
                self.click_pts = np.array([top_left.tolist(), bottom_left.tolist(),
                                    top_right.tolist(), bottom_right.tolist()], dtype='int')

                cv2.destroyWindow('first_image')
                
        else:
            if Status.direct == 'left':
                self.click_pts = np.array(Status.l_monitor_points)
            else:
                self.click_pts = np.array(Status.r_monitor_points)

        return self.click_pts

    def calc_geometric_transform(self):
        m_monitor = np.array([
            [0, 0, Status.monitor_w, Status.monitor_w], 
            [0, Status.monitor_h, 0, Status.monitor_h],
            [0, 0, 0, 0], 
            [0, 0, 0, 0]
        ])
        c_monitor = np.array([
            self.click_pts[:, 0],
            self.click_pts[:, 1],
            self.click_pts[:, 0] * self.click_pts[:, 1],
            [1, 1, 1, 1]
        ])

        c_monitor_inv = np.linalg.inv(c_monitor)
        T = m_monitor.dot(c_monitor_inv)
        
        return T

    def calc_perspective_transform(self, image):
        # 좌상, 좌하, 우상, 우하
        click_pt = np.float32(self.click_pts)
        screen_coor = np.float32([[0, 0], [0, Status.monitor_h], [Status.monitor_w, 0], [Status.monitor_w, Status.monitor_h]])

        M = cv2.getPerspectiveTransform(click_pt, screen_coor)
        dst_img = cv2.warpPerspective(image, M, (Status.monitor_w, Status.monitor_h))
        return dst_img