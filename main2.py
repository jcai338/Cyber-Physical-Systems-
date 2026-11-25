import cv2
from camera import CameraStream
from media_pipe_hands import HandShapeDetector
from game_logic import determine_winner

cam = CameraStream(camera_index=0)
detector = HandShapeDetector(model_path="hand_landmarker.task", num_hands=4)
window_name = "RPS-1 MediaPipe"

try:
    cam.start()
    print("RPS-1 Running. Press 'q' to exit.")

    while True:
        success, frame = cam.get_frame(mirror=True)
        if not success or frame is None:
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
            continue

        annotated_frame, detections = detector.detect_hands(frame)
        top_team, bottom_team = detector.assign_teams(annotated_frame, detections)

        # Filter out "Unknown" gestures before evaluating a winner
        valid_opponent = [
            hand["gesture"] for hand in top_team
            if hand["gesture"] != "Unknown"
        ]
        valid_player = [
            hand["gesture"] for hand in bottom_team
            if hand["gesture"] != "Unknown"
        ]

        if len(valid_player) == 1 and len(valid_opponent) == 1:
            player_move = valid_player[0]
            opponent_move = valid_opponent[0]
            outcome = determine_winner(player_move, opponent_move)

            if outcome == "player":
                result_text = f"YOU WIN! ({player_move} beats {opponent_move})"
                result_color = (0, 255, 0)
            elif outcome == "opponent":
                result_text = f"OPPONENT WINS! ({opponent_move} beats {player_move})"
                result_color = (0, 0, 255)
            else:
                result_text = f"TIE! ({player_move} vs {opponent_move})"
                result_color = (0, 255, 255)
        else:
            result_text = "Waiting for 1 valid hand each..."
            result_color = (200, 200, 200)

        h, _, _ = annotated_frame.shape
        cv2.putText(annotated_frame, f"Opponent: {top_team}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, result_text, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 2)
        cv2.putText(annotated_frame, f"You: {bottom_team}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    detector.close()
    cam.release()
