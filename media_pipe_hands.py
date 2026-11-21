import math
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize MediaPipe HandLandmarker for 4 hands
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=4,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def classify_rps(landmarks):
    wrist = landmarks[0]
    # Check if Index (8 vs 6), Middle (12 vs 10), Ring (16 vs 14), Pinky (20 vs 18) are extended
    fingers = [
        dist(wrist, landmarks[8]) > dist(wrist, landmarks[6]) * 1.15,   # Index
        dist(wrist, landmarks[12]) > dist(wrist, landmarks[10]) * 1.15, # Middle
        dist(wrist, landmarks[16]) > dist(wrist, landmarks[14]) * 1.15, # Ring
        dist(wrist, landmarks[20]) > dist(wrist, landmarks[18]) * 1.15, # Pinky
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

cap = cv2.VideoCapture(0)
print("MediaPipe 4-Hand RPS Running. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(time.time() * 1000)

    result = detector.detect_for_video(mp_image, timestamp_ms)
    h, w, _ = frame.shape

    if result.hand_landmarks:
        for idx, hand_lms in enumerate(result.hand_landmarks):
            gesture = classify_rps(hand_lms)

            # Draw landmarks and bounding box
            xs = [int(lm.x * w) for lm in hand_lms]
            ys = [int(lm.y * h) for lm in hand_lms]
            x_min, x_max = max(0, min(xs) - 15), min(w, max(xs) + 15)
            y_min, y_max = max(0, min(ys) - 15), min(h, max(ys) + 15)

            for x, y in zip(xs, ys):
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 180, 0), 2)
            cv2.putText(
                frame,
                f"Hand {idx + 1}: {gesture}",
                (x_min, max(20, y_min - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 180, 0),
                2,
            )

    cv2.imshow("MediaPipe 4-Hand RPS", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
