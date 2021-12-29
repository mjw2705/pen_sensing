import cv2
import numpy as np


pts = np.zeros((4, 2), dtype=np.float32)
pts_cnt = 0

def click_event(event, x, y, flags, param):
    global pts_cnt, pts

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(param, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)
        pts[pts_cnt] = [x, y]
        pts_cnt += 1

def click_monitor(image):
    global pts_cnt, pts

    while pts_cnt < 4:
        cv2.imshow('first_image', image)
        cv2.setMouseCallback('first_image', click_event, image)
        cv2.waitKey(1)

    if pts_cnt == 4:
        sm = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        top_left = pts[np.argmin(sm)]
        top_right = pts[np.argmin(diff)]
        bottom_left = pts[np.argmax(diff)]
        bottom_right = pts[np.argmax(sm)]

        # 좌상, 좌하, 우상, 우하
        click_pts = np.array([top_left.tolist(), bottom_left.tolist(),
                              top_right.tolist(), bottom_right.tolist()], dtype='int')

        return click_pts

    return [0, 0, 0, 0]

def get_hand_rect(hand_lms, frame_shape, offset, box_ratio=2):
    if hand_lms is not None:
        means = np.mean(hand_lms, axis=0)

        box_w = (means[0] - np.max(hand_lms, axis=0)[0]) * box_ratio
        box_h = (means[1] - np.max(hand_lms, axis=0)[1]) * box_ratio / 2
        sx = max(int((means[0] - box_w) * frame_shape[0] + offset[0]), 0)
        sy = max(int((means[1] - box_h) * frame_shape[1] + offset[1]), 0)
        ex = min(int((means[0] + box_w) * frame_shape[0] + offset[0]), frame_shape[0])
        ey = min(int((means[1] + box_h) * frame_shape[1] + offset[1]), frame_shape[1])

        return (sx, sy), (ex, ey)

        # # min-max로 계산
        # max_value = np.max(hand_lms, axis=0)
        # min_value = np.min(hand_lms, axis=0)
        #
        # sx = max(int(min_value[0] * frame_shape[0]), 0)
        # ex = min(int(max_value[0] * frame_shape[0]), frame_shape[0])
        # sy = max(int(min_value[1] * frame_shape[1]), 0)
        # ey = min(int(max_value[1] * frame_shape[1]), frame_shape[1])
        #
        # return (sx, sy), (ex, ey)

    return (0, 0), (0, 0)

def draw_roi(point, image):
    x1, y1 = point[0]
    x2, y2 = point[1]
    x3, y3 = point[2]
    x4, y4 = point[3]
    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    cv2.line(image, (int(x1), int(y1)), (int(x3), int(y3)), (0, 0, 255), 2)
    cv2.line(image, (int(x4), int(y4)), (int(x3), int(y3)), (0, 0, 255), 2)
    cv2.line(image, (int(x4), int(y4)), (int(x2), int(y2)), (0, 0, 255), 2)

    return point[0], point[1], point[2], point[3]

def hands(hand, image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hand.process(image)
    image.flags.writeable = True

    if results.multi_hand_landmarks:
        hand_lms = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                    for landmark in results.multi_hand_landmarks[0].landmark])
        return hand_lms
    return None

def holistics(holistic, image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image.flags.writeable = True

    pose_lms = None
    if results.pose_landmarks:
        pose_lms = [[landmark.x, landmark.y, landmark.z, landmark.visibility]
                    for landmark in results.pose_landmarks.landmark]

    if results.left_hand_landmarks:
        left_hand_lms = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                    for landmark in results.left_hand_landmarks.landmark])
        return left_hand_lms, pose_lms

    elif results.right_hand_landmarks:
        right_hand_lms = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                    for landmark in results.right_hand_landmarks.landmark])
        return right_hand_lms, pose_lms

    return None, pose_lms