from ultralytics import YOLO
from ultralytics.engine.results import Results

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
        results = iter(self.model(frame, verbose=False))
        result = next(results)
        if not isinstance(result, Results):
            raise TypeError("YOLO inference did not return detection results.")

        # Plot the detection bounding boxes, classes, and confidence scores onto the frame
        annotated_frame = result.plot()

        # Extract useful detection details if needed for game logic later
        # result.boxes contains coordinates, confidence, and class IDs
        detections = result.boxes

        return annotated_frame, detections

    