import cv2
import time
import numpy as np
import sys
import math
import os
from modules.util import *
from modules.click import ClickManager
from modules.hand import HandDetector
from modules.holistic import HolisticDetector
from modules.pen import Pen

'''pdf to img가 필요하면'''
# from pdf2image import convert_from_path

# pdf_path = './ex.pdf'
# pdf_name = pdf_path.split('/')[-1].split('.pdf')[0]
# images = convert_from_path(pdf_path)

# # image save to jpg
# for i, image in enumerate(images):
#     image.save(f'./temps/{pdf_name}_{i}.jpg')

# # image to pdf
# mk_pdf = []
# images[0] = images[0].convert('RGB')
# for image in images[1:]:
#     img = image.convert('RGB')
#     mk_pdf.append(img)
# images[0].save(f'{pdf_name}_drawing.pdf', save_all=True, append_images=mk_pdf)

def main():
    rootPath = os.path.dirname(os.path.abspath(__file__))
    Config.load_config('config.json')
    click = ClickManager()
    hands = HandDetector()
    holistic = HolisticDetector()
    pen = Pen()
    P_x, P_y = 0, 0

    if Status.use_video == Status.use_camera:
        sys.exit("video and camera can't run concurrently")

    if Status.use_video:
        abs_videoPath = rootPath + '/' + Status.video_path
        cap = cv2.VideoCapture(abs_videoPath)
    elif Status.use_camera:
        cap = cv2.VideoCapture(Status.camera_id, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Status.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Status.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    fps = cap.get(cv2.CAP_PROP_FPS)
    # delay = round(1000/fps)

    click_pts = click.click_monitor(cap)
    print("모니터 모서리 좌표: ", click_pts.tolist())
    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(f'ex.avi', fourcc, 30, (Status.monitor_w, Status.monitor_h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            break
        image = cv2.resize(frame, (Status.width, Status.height))
        dst_img = click.calc_perspective_transform(image)
        
        is_hand = hands.process(dst_img)
        # is_hand = holistic.process(dst_img)

        if not is_hand:
            P_x, P_y = 0, 0
            pass
        else:
            hand_lms = hands.get_landmarks()
            hand_img, hand_bbox = get_hand_rect(dst_img, hand_lms, dst_img.shape[:2], box_ratio=4)
            mark_hand(dst_img, hand_lms[0])
            
            is_pen = pen.detect_pen(hand_lms)
            
            if is_pen:
                P_x, P_y = pen.locate_pen(dst_img, hand_img, hand_bbox[:2])
            else:
                P_x, P_y = 0, 0

        cv2.putText(dst_img, f'mode:{pen.mode}', (Status.monitor_w-310, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 2)
        cv2.putText(dst_img, f'x:{int(P_x)}, y:{int(P_y)}', (Status.monitor_w-310, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 2)
        out.write(dst_img)
        dst_img = cv2.resize(dst_img, (Status.width, Status.height))
        cv2.imshow('dst', dst_img)
        cv2.imshow('image', image)
        if cv2.waitKey(1) == 27:
                break
    out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()