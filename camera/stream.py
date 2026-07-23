import cv2
import threading
import time
from utils.logger import logger

class CameraStream:
    def __init__(self, config):
        self.width = config.get('resolution_width', 640)
        self.height = config.get('resolution_height', 640)
        self.framerate = config.get('framerate', 30)
        self.picam2 = None
        self.running = False
        self.frame = None
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        """
        Starts the Picamera2 stream for Sony IMX500 on Raspberry Pi 5.
        This runs the capture loop in a separate background thread to decouple
        camera frame rate from processing loop latency, eliminating delay backlog.
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
            
            # Start background capture thread
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            logger.info("Camera capture background thread started.")
            
        except ImportError:
            logger.error("Picamera2 is not installed or accessible. Cannot start camera.")
            raise
        except Exception as e:
            logger.error(f"Failed to start Picamera2: {e}")
            raise
            
    def _capture_loop(self):
        """Continuously reads frames from Picamera2 at full rate."""
        while self.running:
            try:
                if self.picam2:
                    frame_raw = self.picam2.capture_array()
                    # Swap Red and Blue channels to fix color swap bug
                    frame_bgr = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)
                    with self.lock:
                        self.frame = frame_bgr
                else:
                    time.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in camera capture loop: {e}")
                time.sleep(0.01)

    def read_frame(self):
        """Returns the latest cached frame instantly without blocking."""
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
        # Fallback wait if capture thread hasn't filled the first frame
        time.sleep(0.01)
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
        return False, None

    def stop(self):
        """Stops the camera and background capture thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
            logger.info("Picamera2 stopped.")
