import cv2
import cvzone
import time
import numpy as np
import mediapipe as mp

from utils import *


'''cv2 getperspective transform 함수 사용'''

hand = mp.solutions.hands.Hands(static_image_mode=False,
                                max_num_hands=1,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

holistic = mp.solutions.holistic.Holistic(static_image_mode=False,
                                          min_detection_confidence=0.6,
                                          min_tracking_confidence=0.5)

click_chk = input('클릭?: ')

ptime = 0
frame_shape = (852, 480)
screen_w = 1677
screen_h = 966

cap = cv2.VideoCapture('ex3.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_shape[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_shape[1])
cap.set(cv2.CAP_PROP_FPS, 20)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter('step2.avi', fourcc, 30, frame_shape)

# 모니터 좌표 클릭
if click_chk == 'y':
    success, first_image = cap.read()
    click_pts = click_monitor(first_image)

else:
    click_pts = np.array([[122, 108], [115, 331], [509, 85], [509, 347]])
cv2.destroyWindow('first_image')

# 좌상, 좌하, 우상, 우하
screen_coor = np.float32([[0, 0], [0, screen_h], [screen_w, 0], [screen_w, screen_h]])

while cap.isOpened():
    success, image = cap.read()
    # image = cv2.flip(image, 1)
    h, w, _ = image.shape

    if not success:
        print("Ignoring empty camera frame.")
        break
        # continue

    # 좌표 변환
    click_pt = np.float32(click_pts)
    M = cv2.getPerspectiveTransform(click_pt, screen_coor)
    dst = cv2.warpPerspective(image, M, (screen_w, screen_h))

    point1, point2, point3, point4 = draw_roi(click_pts, image)

    screen_roi = image[point1[1]:point4[1], point1[0]:point4[0]].copy()
    roi_h, roi_w = screen_roi.shape[0:2]

    # 손 detect
    hand_lms, pose_lms = holistics(holistic, dst)

    if hand_lms is None:
        pass
    else:
        # dst
        start, end = get_hand_rect(hand_lms, (screen_w, screen_h), offset=[0, 0], box_ratio=4)
        sx, sy = start
        ex, ey = end
        cv2.rectangle(dst, (sx, sy), (ex, ey), (0, 0, 255), 2)

        pen_x, pen_y = hand_lms[0][0:2]
        pen_abs_x, pen_abs_y = int(pen_x * screen_w), int(pen_y * screen_h)

        cv2.putText(image, f"x:{pen_abs_x}, y:{pen_abs_y}", (610, 460), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1)


    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(image, f"fps: {str(int(fps))}", (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('dst', dst)
    cv2.imshow('image', image)
    # out.write(image)

    if cv2.waitKey(1) == 27:
        break

# out.release()
cap.release()
cv2.destroyAllWindows()
