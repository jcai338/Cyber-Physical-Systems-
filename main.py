import cv2
from camera import CameraStream
from cv_processor import HandShapeDetector
from game_logic import determine_winner

def main():
    # 1. Initialize the webcam stream
    cam = CameraStream(camera_index=0)
    cam.start()

    # 2. Initialize the YOLO hand shape detector
    # (Using standard YOLOv8 nano model as a baseline template)
    detector = HandShapeDetector(model_path="hands.pt")

    print("RPS-1 System Running. Press 'q' in the video window to exit.")
    last_moves = None

    try:
        while True:
            # Get frame from camera module
            success, frame = cam.get_frame(mirror=True)
            if not success:
                print("Failed to grab frame. Exiting...")
                break

            # Process frame through YOLO computer vision component
            processed_frame, detections = detector.detect_hands(frame)

            if detections is not None and len(detections) == 2:
                hands = sorted(
                    zip(
                        detections.xywh[:, 0].tolist(),
                        detections.cls.tolist(),
                    ),
                    key=lambda hand: hand[0],
                )
                left_move = detector.model.names[int(hands[0][1])].lower()
                right_move = detector.model.names[int(hands[1][1])].lower()
                moves = (left_move, right_move)

                if moves != last_moves:
                    winner = determine_winner(left_move, right_move)
                    if winner == "tie":
                        print(
                            f"Tie! Left: {left_move.title()}, "
                            f"Right: {right_move.title()}."
                        )
                    else:
                        winning_side = "Left" if winner == "player" else "Right"
                        print(
                            f"{winning_side} wins! Left: {left_move.title()}, "
                            f"Right: {right_move.title()}."
                        )
                    last_moves = moves
            else:
                last_moves = None

            # Display the resulting vision output
            cv2.imshow("RPS-1 - YOLO Vision Feed", processed_frame)

            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Clean up resources safely
        cam.release()

if __name__ == "__main__":
    main()