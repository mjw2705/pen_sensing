import cv2
import mediapipe as mp
from modules.util import Status


class HolisticDetector:
    def __init__(self):
        self.holistic = mp.solutions.holistic.Holistic(static_image_mode=False,
                                                        min_detection_confidence=0.6,
                                                        min_tracking_confidence=0.5)
        self.landmark = None
        self.width = Status.width
        self.height = Status.height

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.holistic.process(rgb)
        if results.face_landmarks is None:
            return False, False, False

        self.landmark, left_results, right_results = None, None, None
        if results.pose_landmarks:
            self.landmark = [[landmark.x, landmark.y, landmark.z, landmark.visibility]
                        for landmark in results.pose_landmarks.landmark]

        if results.left_hand_landmarks:
            left_hand_lms = [[landmark.x, landmark.y, landmark.z, landmark.visibility]
                        for landmark in results.left_hand_landmarks.landmark]
            left_label = 'Left'
            left_results = [left_hand_lms, left_label]

        if results.right_hand_landmarks:
            right_hand_lms = [[landmark.x, landmark.y, landmark.z, landmark.visibility]
                        for landmark in results.right_hand_landmarks.landmark]
            right_label = 'Right'
            right_results = [right_hand_lms, right_label]

        return left_results, right_results, self.landmark

    def get_landmarks(self):
        return self.landmark