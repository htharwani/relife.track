import numpy as np
import cv2
from utils.logger import logger
from contextlib import ExitStack

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
        
        # Set up persistent context for Network Group and InferVStreams
        self.exit_stack = ExitStack()
        
        # Activate network group
        self.activated_network_group = self.network_group.activate(self.network_group_params)
        self.exit_stack.enter_context(self.activated_network_group)
        
        # Create and enter InferVStreams context
        infer_pipeline_ctx = InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params)
        self.infer_pipeline = self.exit_stack.enter_context(infer_pipeline_ctx)
        
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
        infer_results = self.infer_pipeline.infer(input_data)
            
        # 3. Postprocessing
        boxes = []
        try:
            h, w = frame.shape[:2]
            
            # Dynamically inspect outputs to find the bounding boxes tensor
            for out_name, out_data in infer_results.items():
                
                # Hailo NMS output format: [batch][class_id][box_index][ymin, xmin, ymax, xmax, score]
                if isinstance(out_data, list) and len(out_data) > 0:
                    batch_data = out_data[0]
                    if isinstance(batch_data, list):
                        is_nms_format = False
                        for class_id, class_boxes in enumerate(batch_data):
                            if isinstance(class_boxes, list) or isinstance(class_boxes, np.ndarray):
                                is_nms_format = True
                                for box in class_boxes:
                                    if len(box) >= 5:
                                        ymin, xmin, ymax, xmax, score = box[:5]
                                        
                                        if score < self.conf_threshold:
                                            continue
                                            
                                        x1 = int(xmin * w) if xmin <= 1.0 else int(xmin * (w / input_w))
                                        y1 = int(ymin * h) if ymin <= 1.0 else int(ymin * (h / input_h))
                                        x2 = int(xmax * w) if xmax <= 1.0 else int(xmax * (w / input_w))
                                        y2 = int(ymax * h) if ymax <= 1.0 else int(ymax * (h / input_h))
                                        
                                        boxes.append([x1, y1, x2, y2, score, class_id])
                        
                        # If we confirmed it's the nested NMS format (e.g. 80 classes), stop searching
                        if is_nms_format or len(batch_data) == 80:
                            break
                            
            else:
                shapes = {k: type(v) for k, v in infer_results.items()}
                logger.warning(f"Could not find valid bounding box structure. Available outputs: {shapes}")
                
        except Exception as e:
            logger.error(f"Error decoding YOLO output: {e}")
            
        return np.array(boxes) if boxes else np.empty((0, 6))

    def __del__(self):
        if hasattr(self, 'exit_stack'):
            self.exit_stack.close()
        if hasattr(self, 'target'):
            self.target.release()
