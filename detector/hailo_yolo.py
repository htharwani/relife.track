import numpy as np
import cv2
from utils.logger import logger

class HailoYOLODetector:
    def __init__(self, hef_path, conf_threshold=0.4):
        self.hef_path = hef_path
        self.conf_threshold = conf_threshold
        logger.info(f"Initializing Hailo YOLO detector with {self.hef_path}")
        self._load_model()

    def _load_model(self):
        """Initializes the HailoRT inference session."""
        if not self.hef_path:
            raise ValueError("HEF path for YOLO is empty.")
            
        try:
            from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams, 
                                      InputVStreamParams, OutputVStreamParams, InferVStreams, FormatType)
        except ImportError:
            logger.error("hailo_platform is not installed. Please install HailoRT Python API.")
            return

        self.hef = HEF(self.hef_path)
        self.target = VDevice()
        
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_groups = self.target.configure(self.hef, configure_params)
        self.network_group = self.network_groups[0]
        self.network_group_params = self.network_group.create_params()
        
        # Create input and output stream parameters using proper FormatType Enum
        self.input_vstreams_params = InputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=FormatType.FLOAT32)
        self.output_vstreams_params = OutputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=FormatType.FLOAT32)
        
        self.infer_pipeline = InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params)
        
        # Get input shape info (usually 640x640)
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.input_name = self.input_vstream_info.name
        self.input_shape = self.input_vstream_info.shape
        
        logger.info(f"YOLO Model loaded successfully on Hailo. Input shape: {self.input_shape}")

    def detect(self, frame):
        """
        Runs inference on the frame and returns bounding boxes.
        Returns: numpy array of [x1, y1, x2, y2, score, class_id]
        """
        if not hasattr(self, 'infer_pipeline'):
            return np.empty((0, 6))
            
        # 1. Preprocessing
        if len(self.input_shape) == 4:
            input_h, input_w = self.input_shape[1], self.input_shape[2]
        else:
            input_h, input_w = self.input_shape[0], self.input_shape[1]
            
        resized = cv2.resize(frame, (input_w, input_h))
        # Ensure array is contiguous and formatted as expected by Hailo
        input_data = {self.input_name: np.expand_dims(resized, axis=0).astype(np.float32)}
        
        # 2. Inference
        with self.infer_pipeline as pipeline:
            infer_results = pipeline.infer(input_data)
            
        # 3. Postprocessing
        boxes = []
        try:
            # Note: The output format heavily depends on whether your HEF includes NMS.
            # Assuming it includes NMS and outputs a single tensor of bounding boxes.
            output_name = list(infer_results.keys())[0]
            raw_output = infer_results[output_name][0]
            
            h, w = frame.shape[:2]
            
            # Simple decoder assuming [ymin, xmin, ymax, xmax, score, class_id] normalized between 0-1
            if len(raw_output.shape) == 2 and raw_output.shape[1] >= 5:
                for row in raw_output:
                    ymin, xmin, ymax, xmax, score = row[:5]
                    
                    if score < self.conf_threshold:
                        continue
                        
                    # Some Hailo models output absolute coords, some normalized.
                    # We assume normalized here. If absolute, remove the `* w` and `* h`.
                    x1 = int(xmin * w) if xmin <= 1.0 else int(xmin * (w / input_w))
                    y1 = int(ymin * h) if ymin <= 1.0 else int(ymin * (h / input_h))
                    x2 = int(xmax * w) if xmax <= 1.0 else int(xmax * (w / input_w))
                    y2 = int(ymax * h) if ymax <= 1.0 else int(ymax * (h / input_h))
                    class_id = int(row[5]) if len(row) > 5 else 0
                    
                    boxes.append([x1, y1, x2, y2, score, class_id])
            else:
                # If the shape is completely different, you need custom NMS logic here.
                pass
                
        except Exception as e:
            logger.error(f"Error decoding YOLO output: {e}")
            
        return np.array(boxes) if boxes else np.empty((0, 6))

    def __del__(self):
        if hasattr(self, 'target'):
            self.target.release()
