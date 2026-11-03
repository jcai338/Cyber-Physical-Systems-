from ultralytics import YOLO
import cv2

class HandShapeDetector:
    """
    Handles computer vision hand recognition using YOLO for the RPS-1 system.
    """
    def __init__(self, model_path="yolov8n.pt"):
        """
        Initializes the YOLO model. 
        Note: Replace 'yolov8n.pt' with your custom trained weights path 
        (e.g., 'rps_model.pt') once you train it to recognize Rock, Paper, Scissors.
        """
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)

    def detect_hands(self, frame):
        """
        Processes a video frame through the YOLO model.
        
        Args:
            frame: The OpenCV video frame (numpy array).
            
        Returns:
            tuple: (annotated_frame, detections_data)
        """
        # Run YOLO inference on the frame (verbose=False suppresses excessive console logs)
        results = self.model(frame, verbose=False)

        # Plot the detection bounding boxes, classes, and confidence scores onto the frame
        annotated_frame = results[0].plot()

        # Extract useful detection details if needed for game logic later
        # results[0].boxes contains coordinates, confidence, and class IDs
        detections = results[0].boxes

        return annotated_frame, detections