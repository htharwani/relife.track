import cv2
from utils.logger import logger

class CameraStream:
    def __init__(self, config):
        self.width = config.get('resolution_width', 640)
        self.height = config.get('resolution_height', 640)
        self.framerate = config.get('framerate', 30)
        self.cap = None

    def start(self):
        """
        Starts the libcamera GStreamer pipeline for Sony IMX500 on Raspberry Pi 5.
        Uses libcamerasrc for native performance.
        """
        gst_pipeline = (
            f"libcamerasrc ! "
            f"video/x-raw, width={self.width}, height={self.height}, framerate={self.framerate}/1 ! "
            f"videoconvert ! appsink"
        )
        logger.info(f"Starting camera with pipeline: {gst_pipeline}")
        
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            logger.warning("GStreamer pipeline failed. Falling back to V4L2 (/dev/video0)")
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.framerate)
            
            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera stream.")

    def read_frame(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def stop(self):
        if self.cap:
            self.cap.release()
