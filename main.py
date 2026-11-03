import cv2
from camera import CameraStream
from cv_processor import HandShapeDetector

def main():
    # 1. Initialize the webcam stream
    cam = CameraStream(camera_index=0)
    cam.start()

    # 2. Initialize the YOLO hand shape detector
    # (Using standard YOLOv8 nano model as a baseline template)
    detector = HandShapeDetector(model_path="yolov8n.pt")

    print("RPS-1 System Running. Press 'q' in the video window to exit.")

    try:
        while True:
            # Get frame from camera module
            success, frame = cam.get_frame(mirror=True)
            if not success:
                print("Failed to grab frame. Exiting...")
                break

            # Process frame through YOLO computer vision component
            processed_frame, detections = detector.detect_hands(frame)

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