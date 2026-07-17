import numpy as np
import cv2
from utils.logger import logger
from contextlib import ExitStack

class RepVGGReID:
    def __init__(self, hef_path, device_manager=None):
        self.hef_path = hef_path
        self.device_manager = device_manager
        self.is_mock = True
        logger.info(f"Initializing RepVGG ReID with {self.hef_path}")
        self._load_model()

    def _load_model(self):
        if not self.hef_path:
            raise ValueError("HEF path for RepVGG ReID is empty.")
            
        try:
            from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams, 
                                       InputVStreamParams, OutputVStreamParams, InferVStreams, FormatType)
        except ImportError:
            logger.warning("hailo_platform is not installed. RepVGG ReID running in simulation mode.")
            self.is_mock = True
            return

        try:
            self.hef = HEF(self.hef_path)
            
            # Use shared VDevice target if provided
            if self.device_manager and self.device_manager.device is not None:
                self.target = self.device_manager.device
                logger.info("Using shared VDevice context in RepVGG ReID.")
            else:
                self.target = VDevice()
                logger.warning("No shared VDevice context provided. Initializing standalone VDevice for RepVGG ReID.")
            
            configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
            self.network_groups = self.target.configure(self.hef, configure_params)
            self.network_group = self.network_groups[0]
            self.network_group_params = self.network_group.create_params()
            
            self.input_vstreams_params = InputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=FormatType.FLOAT32)
            self.output_vstreams_params = OutputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=FormatType.FLOAT32)
            
            # Input info
            self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
            self.input_name = self.input_vstream_info.name
            self.input_shape = self.input_vstream_info.shape
            
            # Output info
            self.output_vstream_infos = self.hef.get_output_vstream_infos()
            logger.info(f"RepVGG ReID Model loaded successfully on Hailo. Input shape: {self.input_shape}")
            
            self.is_mock = False
        except Exception as e:
            logger.error(f"Error loading RepVGG ReID HEF model: {e}. Falling back to simulation mode.")
            self.is_mock = True

    def extract(self, person_crop):
        """
        Extracts a 512-dimensional embedding from a person's body crop.
        Returns: np.ndarray of shape (512,)
        """
        if self.is_mock:
            return self._extract_mock(person_crop)

        try:
            # 1. Preprocessing
            input_h, input_w = self.input_shape[1], self.input_shape[2]
            resized = cv2.resize(person_crop, (input_w, input_h))
            input_data = {self.input_name: np.expand_dims(resized, axis=0).astype(np.float32)}
            
            # 2. Inference (dynamic activation)
            from hailo_platform import InferVStreams
            with self.network_group.activate(self.network_group_params):
                with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
                    infer_results = infer_pipeline.infer(input_data)
            
            # 3. Extract output vector
            out_name = self.output_vstream_infos[0].name
            embedding = infer_results[out_name][0].flatten() # Shape (512,)
            return embedding
        except Exception as e:
            logger.error(f"Error in real RepVGG ReID inference: {e}")
            return self._extract_mock(person_crop)

    def _extract_mock(self, person_crop):
        import cv2
        try:
            # Resize to 1x1 to get the average color of the crop
            small = cv2.resize(person_crop, (1, 1))
            # Extract the 3 color channels (BGR)
            feat = small[0, 0].astype(np.float32) / 255.0
            # L2 normalize the feature vector
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            # Pad with zeros to fit 512 dimensions
            embedding = np.zeros(512, dtype=np.float32)
            embedding[:3] = feat
        except Exception:
            embedding = np.zeros(512, dtype=np.float32)
        return embedding

    def __del__(self):
        if hasattr(self, 'exit_stack'):
            self.exit_stack.close()
        if hasattr(self, 'target') and (not self.device_manager or self.device_manager.device is None):
            try:
                self.target.release()
            except Exception as e:
                logger.error(f"Error releasing standalone RepVGG ReID VDevice: {e}")
