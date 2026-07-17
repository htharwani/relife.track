import cv2
from utils.logger import logger

class CameraStream:
    def __init__(self, config):
        self.width = config.get('resolution_width', 640)
        self.height = config.get('resolution_height', 640)
        self.framerate = config.get('framerate', 30)
        self.picam2 = None

    def start(self):
        """
        Starts the Picamera2 stream for Sony IMX500 on Raspberry Pi 5.
        This bypasses OpenCV's backend, which lacks GStreamer support when installed via pip.
        """
        try:
            from picamera2 import Picamera2
            logger.info("Initializing Picamera2...")
            self.picam2 = Picamera2()
            
            # Configure the camera for video stream
            config = self.picam2.create_video_configuration(main={"size": (self.width, self.height), "format": "BGR888"})
            self.picam2.configure(config)
            
            self.picam2.start()
            logger.info("Picamera2 stream started successfully.")
            
        except ImportError:
            logger.error("Picamera2 is not installed or accessible. Cannot start camera.")
            raise
        except Exception as e:
            logger.error(f"Failed to start Picamera2: {e}")
            raise
 
    def read_frame(self):
        if self.picam2 is None:
            return False, None
        try:
            # Capture the latest frame array
            frame_raw = self.picam2.capture_array()
            # Swap Red and Blue channels to fix the color swap bug
            frame_bgr = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)
            return True, frame_bgr
        except Exception as e:
            logger.error(f"Error reading frame from Picamera2: {e}")
            return False, None

    def stop(self):
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
            logger.info("Picamera2 stopped.")
