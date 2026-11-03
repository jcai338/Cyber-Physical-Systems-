
import cv2

class CameraStream:
    """
    Handles webcam initialization, frame retrieval, and safe release
    for the RPS-1 software system.
    """
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None

    def start(self):
        """Initializes the video capture object."""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Error: Could not open webcam at index {self.camera_index}. "
                "Please check if another application is using the camera or try a different index."
            )
        
        # Optional: Set camera resolution properties if needed
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print(f"Webcam initialized successfully at index {self.camera_index}.")

    def get_frame(self, mirror=True):
        """
        Captures a single frame from the webcam stream.
        
        Args:
            mirror (bool): If True, horizontally flips the frame for a natural mirror view.
            
        Returns:
            tuple: (success_boolean, processed_frame)
        """
        if self.cap is None or not self.cap.isOpened():
            print("Error: Camera is not initialized.")
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to grab frame from webcam.")
            return False, None

        # Flip horizontally so left-to-right movement feels natural to the user
        if mirror:
            frame = cv2.flip(frame, 1)

        return True, frame

    def release(self):
        """Safely releases the webcam resource and closes any active windows."""
        if self.cap is not None:
            self.cap.release()
            print("Webcam released and resources cleaned up.")
        cv2.destroyAllWindows()


# --- Standalone Test Block ---
if __name__ == "__main__":
    # Quick test to verify camera.py works independently
    cam = CameraStream(camera_index=0)
    window_name = "Camera Test - RPS-1"
    try:
        cam.start()
        print("Streaming active. Press 'q' or close the video window to quit.")
        
        while True:
            success, frame = cam.get_frame(mirror=True)
            if not success:
                break

            # Display the live stream
            cv2.imshow(window_name, frame)

            # Exit when 'q' is pressed or the user closes the window.
            if (
                cv2.waitKey(1) & 0xFF == ord('q')
                or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
            ):
                break
                
    finally:
        cam.release()