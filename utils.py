import cv2
import numpy as np


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
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.line(image, (x1, y1), (x3, y3), (0, 0, 255), 2)
    cv2.line(image, (x4, y4), (x3, y3), (0, 0, 255), 2)
    cv2.line(image, (x4, y4), (x2, y2), (0, 0, 255), 2)

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