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
            
            # 1. Preprocessing with letterboxing to preserve aspect ratio
            input_h = self.input_shape[0]
            input_w = self.input_shape[1]
            
            # Calculate resize scale and padding values
            r = min(input_w / w_orig, input_h / h_orig)
            new_unpad = int(round(w_orig * r)), int(round(h_orig * r))
            pad_x = (input_w - new_unpad[0]) // 2
            pad_y = (input_h - new_unpad[1]) // 2
            
            if (w_orig, h_orig) != new_unpad:
                resized = cv2.resize(img_crop, new_unpad, interpolation=cv2.INTER_LINEAR)
            else:
                resized = img_crop.copy()
                
            padded = cv2.copyMakeBorder(
                resized,
                pad_y, input_h - new_unpad[1] - pad_y,
                pad_x, input_w - new_unpad[0] - pad_x,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114)
            )
            input_data = {self.input_name: np.expand_dims(padded, axis=0).astype(np.float32)}
            
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
                
                # Apply sigmoid where scores are raw logits (vectorized)
                scores = score_map.copy()
                neg_mask = scores < 0
                if np.any(neg_mask):
                    scores[neg_mask] = 1.0 / (1.0 + np.exp(-scores[neg_mask]))
                
                # Fast numpy filtering
                y_indices, x_indices, a_indices = np.where(scores >= self.conf_threshold)
                
                if len(y_indices) == 0:
                    continue
                    
                for y, x, a in zip(y_indices, x_indices, a_indices):
                    score = scores[y, x, a]
                    
                    idx = a * 4
                    dist = bbox_map[y, x, idx:idx+4]
                    
                    anchor_x = x * stride
                    anchor_y = y * stride
                    
                    x1 = anchor_x - dist[0] * stride
                    y1 = anchor_y - dist[1] * stride
                    x2 = anchor_x + dist[2] * stride
                    y2 = anchor_y + dist[3] * stride
                    
                    # Subtract padding and divide by scale ratio 'r'
                    fx1 = (x1 - pad_x) / r
                    fy1 = (y1 - pad_y) / r
                    fx2 = (x2 - pad_x) / r
                    fy2 = (y2 - pad_y) / r
                    
                    # Clip coordinates to original image crop boundaries
                    fx1 = max(0, min(w_orig, fx1))
                    fy1 = max(0, min(h_orig, fy1))
                    fx2 = max(0, min(w_orig, fx2))
                    fy2 = max(0, min(h_orig, fy2))
                    
                    landmarks = []
                    if kps_map is not None:
                        kps_idx = a * 10
                        kps_dist = kps_map[y, x, kps_idx:kps_idx+10]
                        for k in range(5):
                            kp_x = anchor_x + kps_dist[k*2] * stride
                            kp_y = anchor_y + kps_dist[k*2+1] * stride
                            
                            kp_x_orig = (kp_x - pad_x) / r
                            kp_y_orig = (kp_y - pad_y) / r
                            
                            landmarks.append([
                                max(0, min(w_orig, kp_x_orig)),
                                max(0, min(h_orig, kp_y_orig)),
                                0.99
                            ])
                            
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
