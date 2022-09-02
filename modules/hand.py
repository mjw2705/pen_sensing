import cv2
import numpy as np
import mediapipe as mp
from modules.util import Status


class HandDetector:
    def __init__(self):
        self.hand = mp.solutions.hands.Hands(static_image_mode=False,
                                            max_num_hands=1,
                                            min_detection_confidence=0.5,
                                            min_tracking_confidence=0.5)
        self.landmark = None
        self.width = Status.width
        self.height = Status.height

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hand.process(rgb)
        if results.multi_hand_landmarks is None:
            return False
        self.landmark = [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.multi_hand_landmarks[0].landmark]
        return True

    def get_landmarks(self):
        return self.landmark

    def get_abs_landmarks(self):
        if self.landmark:
            abs_lms = np.array(self.landmark)
            abs_lms[:, 0] = abs_lms[:, 0] * self.width
            abs_lms[:, 1] = abs_lms[:, 1] * self.height
            return abs_lms[:, :2]
        else:
            return None

