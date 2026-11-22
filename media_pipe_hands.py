import math
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandShapeDetector:
    """
    Handles computer vision hand recognition and team assignment using MediaPipe
    for the RPS-1 system.
    """
    def __init__(self, model_path="hand_landmarker.task", num_hands=4):
        print(f"Loading MediaPipe HandLandmarker from {model_path}...")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.last_timestamp_ms = 0

    def _dist(self, a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def _classify_rps(self, landmarks):
        wrist = landmarks[0]
        fingers = [
            self._dist(wrist, landmarks[8]) > self._dist(wrist, landmarks[6]) * 1.15,   # Index
            self._dist(wrist, landmarks[12]) > self._dist(wrist, landmarks[10]) * 1.15, # Middle
            self._dist(wrist, landmarks[16]) > self._dist(wrist, landmarks[14]) * 1.15, # Ring
            self._dist(wrist, landmarks[20]) > self._dist(wrist, landmarks[18]) * 1.15, # Pinky
        ]
        index_ext, middle_ext, ring_ext, pinky_ext = fingers
        extended_count = sum(fingers)

        if extended_count == 0:
            return "Rock"
        elif index_ext and middle_ext and not ring_ext and not pinky_ext:
            return "Scissors"
        elif extended_count >= 4:
            return "Paper"
        return "Unknown"

    def detect_hands(self, frame):
        """
        Detects hands in the frame and classifies each as Rock, Paper, or Scissors.

        Returns:
            tuple: (annotated_frame, detections_list)
        """
        annotated_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Ensure timestamps strictly increase for MediaPipe VIDEO mode
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms

        result = self.detector.detect_for_video(mp_image, timestamp_ms)
        h, w, _ = annotated_frame.shape
        detections = []

        if result.hand_landmarks:
            for idx, hand_lms in enumerate(result.hand_landmarks):
                gesture = self._classify_rps(hand_lms)

                xs = [int(lm.x * w) for lm in hand_lms]
                ys = [int(lm.y * h) for lm in hand_lms]
                x_min, x_max = max(0, min(xs) - 15), min(w, max(xs) + 15)
                y_min, y_max = max(0, min(ys) - 15), min(h, max(ys) + 15)

                for x, y in zip(xs, ys):
                    cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)

                cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (255, 180, 0), 2)
                cv2.putText(
                    annotated_frame,
                    f"Hand {idx + 1}: {gesture}",
                    (x_min, max(20, y_min - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 180, 0),
                    2,
                )

                detections.append({
                    "gesture": gesture,
                    "bbox": (x_min, y_min, x_max, y_max),
                })

        return annotated_frame, detections

    def assign_teams(self, frame, detections):
        """
        Draws a horizontal divider and sorts detected hand gestures into top and bottom teams.
        """
        h, w, _ = frame.shape
        mid_y = h // 2

        cv2.line(frame, (0, mid_y), (w, mid_y), (0, 255, 255), 2)
        cv2.putText(frame, "OPPONENT (TOP)", (20, mid_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "YOU (BOTTOM)", (20, mid_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        top_team = []
        bottom_team = []

        for hand in detections:
            x_min, y_min, x_max, y_max = hand["bbox"]
            center_y = (y_min + y_max) // 2

            if center_y < mid_y:
                top_team.append(hand["gesture"])
            else:
                bottom_team.append(hand["gesture"])

        return top_team, bottom_team

    def close(self):
        self.detector.close()
