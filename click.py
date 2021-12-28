import cv2
import time
import numpy as np
import mediapipe as mp

from utils import *


pts = np.zeros((4, 2), dtype=np.float32)
pts_cnt = 0

def click_event(event, x, y, flags, param):
    global pts_cnt

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(param, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)
        pts[pts_cnt] = [x, y]
        pts_cnt += 1

def main():
    global pts_cnt

    frame_shape = (852, 480)

    image = cv2.imread('img1.jpg')
    image = cv2.resize(image, frame_shape)

    while pts_cnt < 4:
        cv2.imshow('image', image)
        cv2.setMouseCallback('image', click_event, image)
        cv2.waitKey(1)

    if pts_cnt == 4:
        sm = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        top_left = pts[np.argmin(sm)]
        top_right = pts[np.argmin(diff)]
        bottom_left = pts[np.argmax(diff)]
        bottom_right = pts[np.argmax(sm)]

        pts1 = [top_left.tolist(), bottom_left.tolist(), top_right.tolist(), bottom_right.tolist()]
        print(pts1)
        point1, point2, point3, point4 = draw_roi(pts1, image)


    cv2.imshow('image', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


