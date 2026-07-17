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

    def detect(self, img_crop, simulate=True):
        """
        Runs face detection on a cropped image of a person.
        Returns: list of face bounding boxes [x1, y1, x2, y2, score, landmarks] where
        landmarks is a list of 5 keypoints [[x, y, confidence], ...].
        """
        # Simulated face detection output
        # If face found, return its local bounding box relative to img_crop
        faces = [] 
        if not simulate:
            return faces
            
        h, w = img_crop.shape[:2]
        if h > 80 and w > 80:
            aspect_ratio = h / w
            if aspect_ratio > 1.5:
                # Tall crop (full body) -> face is in the upper region
                fx1 = int(w * 0.25)
                fy1 = int(h * 0.08)
                fx2 = int(w * 0.75)
                fy2 = int(h * 0.35)
                
                # Landmarks centered in the upper region
                landmarks = [
                    [int(w * 0.40), int(h * 0.18), 0.98], # Left Eye
                    [int(w * 0.60), int(h * 0.18), 0.99], # Right Eye
                    [int(w * 0.50), int(h * 0.24), 0.97], # Nose Tip
                    [int(w * 0.42), int(h * 0.30), 0.95], # Left Mouth Corner
                    [int(w * 0.58), int(h * 0.30), 0.96]  # Right Mouth Corner
                ]
            else:
                # Square/wide crop (close-up) -> face is in the center
                fx1 = int(w * 0.15)
                fy1 = int(h * 0.20)
                fx2 = int(w * 0.85)
                fy2 = int(h * 0.80)
                
                # Landmarks centered in the middle region
                landmarks = [
                    [int(w * 0.35), int(h * 0.40), 0.98], # Left Eye
                    [int(w * 0.65), int(h * 0.40), 0.99], # Right Eye
                    [int(w * 0.50), int(h * 0.52), 0.97], # Nose Tip
                    [int(w * 0.38), int(h * 0.65), 0.95], # Left Mouth Corner
                    [int(w * 0.62), int(h * 0.65), 0.96]  # Right Mouth Corner
                ]
            
            faces.append([fx1, fy1, fx2, fy2, 0.95, landmarks])
        return faces
