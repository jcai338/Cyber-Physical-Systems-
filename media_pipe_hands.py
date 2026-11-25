import math
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandShapeDetector:
    """
    Handles computer vision hand recognition and team assignment using MediaPipe
    for the RPS-1 system. Supports both VIDEO (webcam) and IMAGE (static file) modes.
    """

    def __init__(self, model_path="hand_landmarker.task", num_hands=4, running_mode="VIDEO"):
        mode = vision.RunningMode.VIDEO if running_mode.upper() == "VIDEO" else vision.RunningMode.IMAGE
        self.running_mode = mode

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mode,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.last_timestamp_ms = 0

    def _dist(self, a, b, w: int, h: int) -> float:
        # Scale by image width/height so distance is rotation-invariant on non-square frames
        return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)

    def _classify_rps(self, landmarks, w: int, h: int) -> str:
        wrist = landmarks[0]
        finger_joints = ((8, 6), (12, 10), (16, 14), (20, 18))
        fingers = [
            self._dist(wrist, landmarks[tip], w, h)
            > self._dist(wrist, landmarks[joint], w, h) * 1.15
            for tip, joint in finger_joints
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
        """Detects hands in a BGR frame and classifies each as Rock, Paper, or Scissors."""
        annotated_frame = frame.copy()
        h, w, _ = annotated_frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        if self.running_mode == vision.RunningMode.VIDEO:
            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= self.last_timestamp_ms:
                timestamp_ms = self.last_timestamp_ms + 1
            self.last_timestamp_ms = timestamp_ms
            result = self.detector.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self.detector.detect(mp_image)

        detections = []
        if result.hand_landmarks:
            for idx, hand_lms in enumerate(result.hand_landmarks):
                gesture = self._classify_rps(hand_lms, w, h)

                xs = [int(lm.x * w) for lm in hand_lms]
                ys = [int(lm.y * h) for lm in hand_lms]
                x_min, x_max = max(0, min(xs) - 15), min(w, max(xs) + 15)
                y_min, y_max = max(0, min(ys) - 15), min(h, max(ys) + 15)

                for x, y in zip(xs, ys):
                    cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)

                cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (255, 180, 0), 2)
                cv2.putText(
                    annotated_frame,
                    f"{gesture}",
                    (x_min, max(20, y_min - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 180, 0),
                    2,
                )

                detections.append({
                    "gesture": gesture,
                })

        return annotated_frame, detections

    def assign_teams(self, frame, detections):
        """
        Sorts detected hands into top (Opponent) and bottom (You) teams,
        ordered left-to-right so index 0 = Left Hand and index 1 = Right Hand.
        """
        h, w, _ = frame.shape
        mid_y = h // 2

        cv2.line(frame, (0, mid_y), (w, mid_y), (0, 255, 255), 2)
        cv2.putText(frame, "OPPONENT (TOP)", (20, mid_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "YOU (BOTTOM)", (20, mid_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        top_hands = [d for d in detections if d["center_y"] < mid_y]
        bottom_hands = [d for d in detections if d["center_y"] >= mid_y]

        # Sort left-to-right by center_x
        top_hands.sort(key=lambda d: d["center_x"])
        bottom_hands.sort(key=lambda d: d["center_x"])

        return top_hands, bottom_hands

    def close(self):
        self.detector.close()

