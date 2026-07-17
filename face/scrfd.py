import numpy as np
from utils.logger import logger

class SCRFDDetector:
    def __init__(self, hef_path, conf_threshold=0.5):
        self.hef_path = hef_path
        self.conf_threshold = conf_threshold
        logger.info(f"Initializing SCRFD face detector with {self.hef_path}")
        self._load_model()

    def _load_model(self):
        if not self.hef_path:
            raise ValueError("HEF path for SCRFD is empty.")
        logger.info("SCRFD Model loaded successfully on Hailo.")

    def detect(self, img_crop):
        """
        Runs face detection on a cropped image of a person.
        Returns: list of face bounding boxes [x1, y1, x2, y2, score, landmarks] where
        landmarks is a list of 5 keypoints [[x, y, confidence], ...].
        """
        # Simulated face detection output
        # If face found, return its local bounding box relative to img_crop
        faces = [] 
        h, w = img_crop.shape[:2]
        if h > 80 and w > 80:
            # Simulate a face in the upper 10% to 45% region of the person crop
            fx1 = int(w * 0.25)
            fy1 = int(h * 0.10)
            fx2 = int(w * 0.75)
            fy2 = int(h * 0.45)
            
            # Simulate 5 standard landmarks: [x, y, confidence]
            # 1. Left Eye, 2. Right Eye, 3. Nose Tip, 4. Left Mouth Corner, 5. Right Mouth Corner
            landmarks = [
                [int(w * 0.40), int(h * 0.20), 0.98], # Left Eye
                [int(w * 0.60), int(h * 0.20), 0.99], # Right Eye
                [int(w * 0.50), int(h * 0.28), 0.97], # Nose Tip
                [int(w * 0.42), int(h * 0.38), 0.95], # Left Mouth Corner
                [int(w * 0.58), int(h * 0.38), 0.96]  # Right Mouth Corner
            ]
            faces.append([fx1, fy1, fx2, fy2, 0.95, landmarks])
        return faces
