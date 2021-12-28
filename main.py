import cv2
import time
import numpy as np
import mediapipe as mp

from utils import *



hand = mp.solutions.hands.Hands(static_image_mode=False,
                                max_num_hands=1,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

holistic = mp.solutions.holistic.Holistic(static_image_mode=False,
                                          min_detection_confidence=0.6,
                                          min_tracking_confidence=0.5)

ptime = 0
frame_shape = (852, 480)
screen_w = 1677
screen_h = 966

# 좌상, 좌하, 우상, 우하
screen_coor = np.float32([[0, 0], [0, screen_h], [screen_w, 0], [screen_w, screen_h]])
img_coor = [[122, 108], [115, 331], [509, 85], [509, 347]]

m_monitor = np.array([[0, 0, screen_w, screen_w], [0, screen_h, 0, screen_h], [0, 0, 0, 0], [0, 0, 0, 0]])
c_monitor = np.array([[122, 115, 509, 509], [108, 331, 85, 347], [122*108, 115*331, 509*85, 509*347], [1, 1, 1, 1]])
c_monitor_inv = np.linalg.inv(c_monitor)
T = m_monitor.dot(c_monitor_inv)


cap = cv2.VideoCapture('ex3.mp4')
# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_shape[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_shape[1])
cap.set(cv2.CAP_PROP_FPS, 20)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('step1.avi', fourcc, 30, frame_shape)


while cap.isOpened():
    success, image = cap.read()
    # image = cv2.flip(image, 1)
    h, w, _ = image.shape

    if not success:
        print("Ignoring empty camera frame.")
        break
        # continue

    point1, point2, point3, point4 = draw_roi(img_coor, image)
    screen_roi = image[point1[1]:point4[1], point1[0]:point4[0]].copy()
    roi_h, roi_w, _ = screen_roi.shape

    # 손 detect
    hand_lms, pose_lms = holistics(holistic, screen_roi)

    if hand_lms is None:
        pass
    else:
        start, end = get_hand_rect(hand_lms, (roi_w, roi_h), offset=point1, box_ratio=4)
        sx, sy = start
        ex, ey = end
        cv2.rectangle(image, (sx, sy), (ex, ey), (0, 0, 255), 2)

        # 펜촉 위치
        pen_x, pen_y = hand_lms[0][0:2]
        pen_abs_x, pen_abs_y = int(pen_x * roi_w + point1[0]), int(pen_y * roi_h + point1[1])

        P_x, P_y, _, _ = T.dot([pen_abs_x, pen_abs_y, 1, 1])
        cv2.circle(image, (pen_abs_x, pen_abs_y), 5, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(image, f"x:{int(P_x)}, y:{int(P_y)}", (620, 450), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1)


    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(image, f"fps: {str(int(fps))}", (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('image', image)
    out.write(image)

    if cv2.waitKey(1) == 27:
        break

out.release()
cap.release()
cv2.destroyAllWindows()
