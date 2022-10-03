import os
import cv2
import numpy as np
import json
import sys



class Status:
    camera_id = 0
    use_camera = True
    use_video = False
    video_path = './video/ex1.mp4'
    width = 640
    height = 480
    save_dir = './output'
    monitor_w = 1677
    monitor_h = 966
    direct = 'left'
    r_monitor_points = [0, 0, 0, 0]
    l_monitor_points = [0, 0, 0, 0]
    click_state = False
    

class Config:
    def load_config(path):
        with open(path, 'r', encoding='utf-8') as file:
            config = json.load(file)
            Status.camera_id = config["camera_id"]
            Status.use_camera = config["use_camera"]
            Status.use_video = config["use_video"]
            Status.video_path = config["video_path"]
            Status.width = config["width"]
            Status.height = config["height"]
            Status.save_dir = config["savedir"]
            Status.monitor_w = config["monitor_width"]
            Status.monitor_h = config["monitor_height"]
            Status.direct = config["direct"]
            Status.r_monitor_points = config["r_monitor_points"]
            Status.l_monitor_points = config["l_monitor_points"]
            Status.click_state = config["click_state"]

        try:
            if not os.path.exists(Status.save_dir):
                os.makedirs(Status.save_dir)
        except OSError:
            print("Error: Failed to create the directory.")


def draw_monitor_roi(point, image):
    # 좌상, 좌하, 우상, 우하
    x1, y1 = point[0]
    x2, y2 = point[1]
    x3, y3 = point[2]
    x4, y4 = point[3]
    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    cv2.line(image, (int(x1), int(y1)), (int(x3), int(y3)), (0, 0, 255), 2)
    cv2.line(image, (int(x4), int(y4)), (int(x3), int(y3)), (0, 0, 255), 2)
    cv2.line(image, (int(x4), int(y4)), (int(x2), int(y2)), (0, 0, 255), 2)

    return point[0], point[1], point[2], point[3]

def screen_roi(click_pts):
    min_scale = np.min(click_pts, axis=0)
    max_scale = np.max(click_pts, axis=0)
    min_x, min_y = min_scale
    max_x, max_y = max_scale
    return min_scale, max_scale

def get_hand_rect(image, hand_lms, frame_shape, box_ratio=2):
    if hand_lms is not None:
        h, w = frame_shape
        cx, cy = hand_lms[2][0], hand_lms[2][1]
        margin = 160
        # box_halfw = abs(cx - hand_lms[4][0]) * 2 * box_ratio
        # box_halfh = abs(cy - hand_lms[6][1])
        sx = max(int((cx * w) - margin), 0)
        sy = max(int((cy * h) - margin * 1.2), 0)
        ex = min(int((cx * w) + margin), w)
        ey = min(int((cy * h) + margin // 1.2), h)

        cv2.rectangle(image, (sx, sy), (ex, ey), (0, 0, 255), 2)
        hand_img = image[sy:ey, sx:ex].copy()
        
        return hand_img, [sx, sy, ex, ey]
    return None, [0, 0, 0, 0]

def mark_hand(image, hand_lms0):
    x, y = hand_lms0[0], hand_lms0[1]
    cv2.circle(image, (int(x * Status.monitor_w), int(y * Status.monitor_h)), 10, (0, 0, 255), -1, cv2.LINE_AA)
