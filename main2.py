import cv2
from camera import CameraStream
from media_pipe_hands import HandShapeDetector

cam = CameraStream(camera_index=0)
detector = HandShapeDetector(model_path="hand_landmarker.task", num_hands=4)
window_name = "RPS-1 MediaPipe"

try:
    cam.start()
    print("RPS-1 Running. Press 'q' to exit.")

    while True:
        success, frame = cam.get_frame(mirror=True)
        if not success or frame is None:
            break

        annotated_frame, detections = detector.detect_hands(frame)
        top_team, bottom_team = detector.assign_teams(annotated_frame, detections)

        h, _, _ = annotated_frame.shape
        cv2.putText(annotated_frame, f"Opponent: {top_team}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"You: {bottom_team}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    detector.close()
    cam.release()
