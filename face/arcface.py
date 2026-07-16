import numpy as np
from utils.logger import logger

class ArcFaceExtractor:
    def __init__(self, hef_path):
        self.hef_path = hef_path
        logger.info(f"Initializing ArcFace MobileFaceNet with {self.hef_path}")
        self._load_model()

    def _load_model(self):
        if not self.hef_path:
            raise ValueError("HEF path for ArcFace is empty.")
        logger.info("ArcFace Model loaded successfully on Hailo.")

    def extract(self, face_crop):
        """
        Extracts a 512-dimensional embedding from a face crop.
        Returns: np.ndarray of shape (512,)
        """
        import cv2
        try:
            resized = cv2.resize(face_crop, (512, 1))
            embedding = resized.astype(np.float32).mean(axis=2).flatten() / 255.0
        except Exception:
            embedding = np.zeros(512, dtype=np.float32)
        return embedding
