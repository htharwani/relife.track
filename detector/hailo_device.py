import os
from utils.logger import logger
from contextlib import ExitStack

class HailoDeviceManager:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(HailoDeviceManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.device = None
        self.exit_stack = ExitStack()
        self._init_device()
        self._initialized = True

    def _init_device(self):
        try:
            from hailo_platform import VDevice
            logger.info("Initializing shared Hailo VDevice...")
            self.device = VDevice()
            logger.info("Shared Hailo VDevice initialized successfully.")
        except ImportError:
            logger.warning("hailo_platform is not installed. Device manager running in simulation mode.")
            self.device = None
        except Exception as e:
            logger.error(f"Failed to initialize Hailo VDevice: {e}")
            self.device = None

    def release(self):
        logger.info("Releasing shared Hailo VDevice resources...")
        try:
            self.exit_stack.close()
        except Exception as e:
            logger.error(f"Error closing exit stack: {e}")
            
        if self.device is not None:
            try:
                self.device.release()
            except Exception as e:
                logger.error(f"Error releasing VDevice: {e}")
            self.device = None
        self._initialized = False
        HailoDeviceManager._instance = None

    def __del__(self):
        self.release()
