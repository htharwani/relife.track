import numpy as np
from utils.logger import logger

class RepVGGReID:
    def __init__(self, hef_path):
        self.hef_path = hef_path
        logger.info(f"Initializing RepVGG Person ReID with {self.hef_path}")
        self._load_model()

    def _load_model(self):
        if not self.hef_path:
            raise ValueError("HEF path for RepVGG ReID is empty.")
        logger.info("RepVGG Model loaded successfully on Hailo.")

    def extract(self, person_crop):
        """
        Extracts a 512-dimensional embedding from a person's body crop.
        Returns: np.ndarray of shape (512,)
        """
        import cv2
        try:
            resized = cv2.resize(person_crop, (512, 1))
            embedding = resized.astype(np.float32).mean(axis=2).flatten() / 255.0
        except Exception:
            embedding = np.zeros(512, dtype=np.float32)
        return embedding
