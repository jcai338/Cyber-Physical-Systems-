"""Play a two-person Rock-Paper-Scissors round using MediaPipe Hands."""

import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from game_logic import determine_winner


def _distance(first, second):
    return math.hypot(first.x - second.x, first.y - second.y)


def classify_rps(landmarks):
    """Classify a hand as rock, paper, scissors, or unknown."""
    wrist = landmarks[0]
    finger_pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
    extended = [
        _distance(wrist, landmarks[tip])
        > _distance(wrist, landmarks[joint]) * 1.15
        for tip, joint in finger_pairs
    ]

    if not any(extended):
        return "rock"
    if extended == [True, True, False, False]:
        return "scissors"
    if all(extended):
        return "paper"
    return "unknown"


def _draw_hand(frame, landmarks, label):
    height, width = frame.shape[:2]
    xs = [int(point.x * width) for point in landmarks]
    ys = [int(point.y * height) for point in landmarks]
    left = max(0, min(xs) - 15)
    right = min(width - 1, max(xs) + 15)
    top = max(0, min(ys) - 15)
    bottom = min(height - 1, max(ys) + 15)

    cv2.rectangle(frame, (left, top), (right, bottom), (255, 180, 0), 2)
    cv2.putText(
        frame,
        label,
        (left, max(25, top - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 180, 0),
        2,
    )


def main():
    model_path = Path(__file__).with_name("hand_landmarker.task")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"MediaPipe model not found: {model_path}. "
            "Place hand_landmarker.task beside main2.py."
        )

    camera = cv2.VideoCapture(0)
    detector = None
    last_moves = None
    round_message = "Show one RPS gesture per player"

    try:
        if not camera.isOpened():
            raise RuntimeError("Could not open webcam at index 0.")

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        detector = vision.HandLandmarker.create_from_options(options)
        print("MediaPipe RPS running. Press 'q' or close the window to quit.")

        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to read a frame from the webcam.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = time.monotonic_ns() // 1_000_000
            result = detector.detect_for_video(image, timestamp_ms)

            hands = []
            for landmarks in result.hand_landmarks or []:
                center_x = sum(point.x for point in landmarks) / len(landmarks)
                hands.append((center_x, landmarks, classify_rps(landmarks)))
            hands.sort(key=lambda hand: hand[0])

            if len(hands) == 2:
                left_move, right_move = hands[0][2], hands[1][2]
                moves = (left_move, right_move)

                if left_move != "unknown" and right_move != "unknown":
                    if moves != last_moves:
                        winner = determine_winner(left_move, right_move)
                        if winner == "tie":
                            round_message = "Tie!"
                        else:
                            side = "Left" if winner == "player" else "Right"
                            round_message = f"{side} player wins!"
                        print(
                            f"{round_message} Left: {left_move.title()}, "
                            f"Right: {right_move.title()}."
                        )
                        last_moves = moves
                else:
                    last_moves = None
                    round_message = "Show clear Rock, Paper, or Scissors gestures"

                _draw_hand(frame, hands[0][1], f"Left: {left_move.title()}")
                _draw_hand(frame, hands[1][1], f"Right: {right_move.title()}")
            else:
                last_moves = None
                round_message = "Show one RPS gesture per player"

            cv2.putText(
                frame,
                round_message,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            window_name = "MediaPipe Rock Paper Scissors"
            cv2.imshow(window_name, frame)
            if (
                cv2.waitKey(1) & 0xFF == ord("q")
                or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
            ):
                break
    finally:
        if detector is not None:
            detector.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()