import numpy as np
import cv2
from utils.logger import logger
from contextlib import ExitStack

class SCRFDDetector:
    def __init__(self, hef_path, device_manager=None, conf_threshold=0.5):
        self.hef_path = hef_path
        self.device_manager = device_manager
        self.conf_threshold = conf_threshold
        self.is_mock = True
        logger.info(f"Initializing SCRFD face detector with {self.hef_path}")
        self._load_model()

    def _load_model(self):
        if not self.hef_path:
            raise ValueError("HEF path for SCRFD is empty.")
        
        try:
            from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams, 
                                       InputVStreamParams, OutputVStreamParams, InferVStreams, FormatType)
        except ImportError:
            logger.warning("hailo_platform is not installed. SCRFD running in simulation mode.")
            self.is_mock = True
            return

        try:
            self.hef = HEF(self.hef_path)
            
            # Use shared target device from the manager if available
            if self.device_manager and self.device_manager.device is not None:
                self.target = self.device_manager.device
                logger.info("Using shared VDevice context in SCRFD detector.")
            else:
                self.target = VDevice()
                logger.warning("No shared VDevice context provided. Initializing standalone VDevice for SCRFD.")
            
            configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
            self.network_groups = self.target.configure(self.hef, configure_params)
            self.network_group = self.network_groups[0]
            self.network_group_params = self.network_group.create_params()
            
            # Create input/output params
            self.input_vstreams_params = InputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=FormatType.FLOAT32)
            self.output_vstreams_params = OutputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=FormatType.FLOAT32)
            
        # Get input shape info
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.input_name = self.input_vstream_info.name
        self.input_shape = self.input_vstream_info.shape
        
        # Get output stream info
        self.output_vstream_infos = self.hef.get_output_vstream_infos()
        logger.info(f"SCRFD Model loaded successfully on Hailo. Input shape: {self.input_shape}")
        
        self.is_mock = False
    except Exception as e:
        logger.error(f"Error loading SCRFD HEF model: {e}. Falling back to simulation mode.")
        self.is_mock = True

    def detect(self, img_crop, simulate=True):
        """
        Runs face detection on a cropped image of a person.
        Returns: list of face bounding boxes [x1, y1, x2, y2, score, landmarks] where
        landmarks is a list of 5 keypoints [[x, y, confidence], ...].
        """
        if self.is_mock or simulate:
            return self._detect_mock(img_crop, simulate)

        try:
            h_orig, w_orig = img_crop.shape[:2]
            
            # 1. Preprocessing
            input_h, input_w = self.input_shape[1], self.input_shape[2]
            resized = cv2.resize(img_crop, (input_w, input_h))
            input_data = {self.input_name: np.expand_dims(resized, axis=0).astype(np.float32)}
            
            # 2. Inference (dynamic activation)
            from hailo_platform import InferVStreams
            with self.network_group.activate(self.network_group_params):
                with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
                    infer_results = infer_pipeline.infer(input_data)
            
            # 3. Postprocessing (decoding outputs dynamically)
            strides = [8, 16, 32]
            outputs_by_stride = {8: {}, 16: {}, 32: {}}
            
            for name, data in infer_results.items():
                shape = data.shape
                h_out, w_out = shape[1], shape[2]
                
                # Determine stride resolution
                stride = None
                for s in strides:
                    if abs(input_h / s - h_out) < 2:
                        stride = s
                        break
                if stride is None:
                    continue
                    
                last_dim = shape[-1]
                if last_dim in [1, 2]:
                    outputs_by_stride[stride]['score'] = data[0]
                elif last_dim in [4, 8]:
                    outputs_by_stride[stride]['bbox'] = data[0]
                elif last_dim in [10, 20]:
                    outputs_by_stride[stride]['kps'] = data[0]
                    
            bboxes = []
            kpss = []
            
            for stride in strides:
                out = outputs_by_stride[stride]
                if 'score' not in out or 'bbox' not in out:
                    continue
                    
                score_map = out['score']
                bbox_map = out['bbox']
                kps_map = out.get('kps', None)
                
                H, W, C = score_map.shape
                num_anchors = C
                
                for y in range(H):
                    for x in range(W):
                        for a in range(num_anchors):
                            score = score_map[y, x, a]
                            # Apply sigmoid if scores are raw logits
                            if score < 0:
                                score = 1.0 / (1.0 + np.exp(-score))
                                
                            if score < self.conf_threshold:
                                continue
                                
                            idx = a * 4
                            dist = bbox_map[y, x, idx:idx+4]
                            
                            anchor_x = x * stride
                            anchor_y = y * stride
                            
                            x1 = anchor_x - dist[0] * stride
                            y1 = anchor_y - dist[1] * stride
                            x2 = anchor_x + dist[2] * stride
                            y2 = anchor_y + dist[3] * stride
                            
                            scale_x = w_orig / input_w
                            scale_y = h_orig / input_h
                            
                            fx1 = max(0, x1 * scale_x)
                            fy1 = max(0, y1 * scale_y)
                            fx2 = min(w_orig, x2 * scale_x)
                            fy2 = min(h_orig, y2 * scale_y)
                            
                            landmarks = []
                            if kps_map is not None:
                                kps_idx = a * 10
                                kps_dist = kps_map[y, x, kps_idx:kps_idx+10]
                                for k in range(5):
                                    kp_x = anchor_x + kps_dist[k*2] * stride
                                    kp_y = anchor_y + kps_dist[k*2+1] * stride
                                    landmarks.append([kp_x * scale_x, kp_y * scale_y, 0.99])
                                    
                            bboxes.append([fx1, fy1, fx2, fy2, score])
                            kpss.append(landmarks)
                            
            if not bboxes:
                return []
                
            # Keep only the face with the highest score
            best_idx = np.argmax([b[4] for b in bboxes])
            return [[bboxes[best_idx][0], bboxes[best_idx][1], bboxes[best_idx][2], bboxes[best_idx][3], bboxes[best_idx][4], kpss[best_idx]]]
            
        except Exception as e:
            logger.error(f"Error running real SCRFD inference: {e}")
            return self._detect_mock(img_crop, simulate)

    def _detect_mock(self, img_crop, simulate=True):
        faces = [] 
        if not simulate:
            return faces
            
        h, w = img_crop.shape[:2]
        if h > 80 and w > 80:
            # Single, highly stable face box covering the upper-middle region
            fx1 = int(w * 0.20)
            fy1 = int(h * 0.15)
            fx2 = int(w * 0.80)
            fy2 = int(h * 0.65)
            
            # Stable landmarks centered in the face region
            landmarks = [
                [int(w * 0.38), int(h * 0.30), 0.98], # Left Eye
                [int(w * 0.62), int(h * 0.30), 0.99], # Right Eye
                [int(w * 0.50), int(h * 0.42), 0.97], # Nose Tip
                [int(w * 0.40), int(h * 0.54), 0.95], # Left Mouth Corner
                [int(w * 0.60), int(h * 0.54), 0.96]  # Right Mouth Corner
            ]
            
            faces.append([fx1, fy1, fx2, fy2, 0.95, landmarks])
        return faces

    def __del__(self):
        if hasattr(self, 'exit_stack'):
            self.exit_stack.close()
        if hasattr(self, 'target') and (not self.device_manager or self.device_manager.device is None):
            try:
                self.target.release()
            except Exception as e:
                logger.error(f"Error releasing standalone SCRFD VDevice: {e}")
