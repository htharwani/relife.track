import cv2
import uuid
import numpy as np
from datetime import datetime
from config.config_manager import ConfigManager
from database.postgres_client import PostgresClient
from vector_db.vector_store import VectorStore
from camera.stream import CameraStream
from detector.hailo_yolo import HailoYOLODetector
from tracker.byte_tracker import ByteTrackerWrapper
from face.scrfd import SCRFDDetector
from face.arcface import ArcFaceExtractor
from reid.repvgg import RepVGGReID
from detector.hailo_device import HailoDeviceManager
from utils.logger import logger
from utils.normalization import normalize_embedding
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time

latest_frame = None
frame_lock = threading.Lock()

def get_iou(box1, box2):
    """Calculates Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2]."""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    return float(inter_area) / union_area

class StreamingHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress request logging to avoid terminal clutter
        return

    def do_GET(self):
        global latest_frame
        if self.path == '/stream':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        frame_data = latest_frame
                    
                    if frame_data is not None:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(frame_data)))
                        self.end_headers()
                        self.wfile.write(frame_data)
                        self.wfile.write(b'\r\n')
                    
                    time.sleep(0.03)  # ~30 FPS
            except Exception:
                pass
        else:
            self.send_response(404)
            self.end_headers()

class UniquePersonCounter:
    def __init__(self, use_db=True, port=5000):
        self.use_db = use_db
        self.stream_port = port
        self.config_manager = ConfigManager()
        self.config = self.config_manager.config
        
        # Init DB and Vector Store
        if self.use_db:
            self.db = PostgresClient(self.config['database'])
        else:
            self.db = None
        self.faiss = VectorStore(dim=512)
        
        # Initialize Shared Hailo Device Manager
        self.device_manager = HailoDeviceManager()
        
        # Init Models
        models_cfg = self.config['models']
        pipeline_cfg = self.config.get('pipeline', {})
        yolo_thresh = pipeline_cfg.get('yolo_threshold', 0.4)
        face_thresh = pipeline_cfg.get('face_threshold', 0.5)
        
        self.detector = HailoYOLODetector(models_cfg['yolo'], device_manager=self.device_manager, conf_threshold=yolo_thresh)
        self.tracker = ByteTrackerWrapper(track_thresh=yolo_thresh)
        self.scrfd = SCRFDDetector(models_cfg['scrfd'], device_manager=self.device_manager, conf_threshold=face_thresh)
        self.arcface = ArcFaceExtractor(models_cfg['arcface'], device_manager=self.device_manager)
        self.reid = RepVGGReID(models_cfg['reid'], device_manager=self.device_manager)
        
        # Init Camera
        self.camera = CameraStream(self.config['camera'])
        
        # State tracking
        self.active_tracks = {} # track_id: visitor_uuid
        self.unique_visitors = set() # set of unique visitor_uuids seen in this session
        self.tracks_with_face = set() # track_ids that have had a face registered
        self.track_face_bbox = {} # track_id: (x1, y1, x2, y2) global coordinates of face
        self.simulate_face = False # Toggle to simulate face visibility
        
        # Hourly statistics
        self.camera_id = self.config['camera'].get('id', 1)
        self.hourly_total_in = 0
        self.hourly_total_out = 0
        self.hourly_peak_occupancy = 0
        self.hourly_occupancy_samples = []
        self.current_hour = datetime.now().hour
        self.current_date = datetime.now().date()
        self.last_db_sync_time = 0
        
        # ReID body cache (visitor_uuid: last_reid_embedding)
        self.reid_history = {}
        
        # Maps UUID to a stable ID for displaying
        self.uuid_to_stable_id = {}
        
        # Load display IDs from database first (for consistency across restarts)
        db_visitors = []
        if self.use_db and self.db:
            try:
                rows = self.db.get_all_visitors()
                for r in rows:
                    u_str = r[0]
                    u = uuid.UUID(str(u_str))
                    db_visitors.append(u)
            except Exception as e:
                logger.warning(f"Could not load visitors from database at startup: {e}")
                
        # Pre-populate self.uuid_to_stable_id using sorted database visitors
        for idx, u in enumerate(db_visitors):
            self.uuid_to_stable_id[u] = idx + 1
            
        # Add any UUIDs from FAISS mapping that are not yet in the map
        for u in self.faiss.uuid_mapping:
            u_obj = uuid.UUID(str(u)) if isinstance(u, str) else u
            if u_obj not in self.uuid_to_stable_id:
                self.uuid_to_stable_id[u_obj] = len(self.uuid_to_stable_id) + 1
                
        # Finally, set the next display ID
        self.next_display_id = len(self.uuid_to_stable_id) + 1
        logger.info(f"Initialized stable display ID mapping: {len(self.uuid_to_stable_id)} profiles loaded. Next ID is {self.next_display_id}")
        
        # Bounding box smoothing cache
        self.smoothed_bboxes = {}
        
        # Tracks timestamps of when each visitor UUID was last active
        self.reid_last_seen = {}
        
        # Stores the best face detection score obtained for each track_id in the current session
        self.track_best_face_score = {}

    def run(self):
        # Start MJPEG HTTP server thread
        try:
            server = HTTPServer(('0.0.0.0', self.stream_port), StreamingHandler)
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            logger.info(f"MJPEG Stream server running at http://localhost:{self.stream_port}/stream")
        except Exception as e:
            logger.error(f"Failed to start MJPEG Stream server: {e}")

        self.camera.start()
        logger.info("Pipeline started.")
        
        try:
            while True:
                ret, frame = self.camera.read_frame()
                if not ret:
                    continue

                # 1. Detection
                all_boxes = self.detector.detect(frame)
                
                # Filter to only track 'person' class (COCO class ID 0)
                if all_boxes.size > 0:
                    person_boxes = all_boxes[all_boxes[:, 5] == 0]
                    valid_boxes = []
                    for box in person_boxes:
                        x1, y1, x2, y2 = box[:4]
                        w = x2 - x1
                        h = y2 - y1
                        # Suppress flat false positives (pedestrians are taller than they are wide)
                        if h >= w * 0.90:
                            valid_boxes.append(box)
                    boxes = np.array(valid_boxes) if valid_boxes else np.empty((0, 6))
                else:
                    boxes = all_boxes
                
                # 2. Tracking
                # Simulated ByteTrack logic mapping
                tracked_objects = self.tracker.update(boxes, frame.shape, frame.shape)
                
                # Sort tracked objects: prioritize those that already have a face registered, then by score
                tracked_objects = sorted(
                    tracked_objects,
                    key=lambda t: (1 if t.track_id in self.tracks_with_face else 0, t.score),
                    reverse=True
                )
                
                # Suppress highly overlapping duplicate tracks (IoU > 0.45)
                suppressed_objects = []
                for track in tracked_objects:
                    x1, y1, x2, y2 = map(int, track.tlbr)
                    box = [x1, y1, x2, y2]
                    
                    is_duplicate = False
                    for kept_track in suppressed_objects:
                        kx1, ky1, kx2, ky2 = map(int, kept_track.tlbr)
                        kbox = [kx1, ky1, kx2, ky2]
                        if get_iou(box, kbox) > 0.45:
                            is_duplicate = True
                            break
                            
                    if not is_duplicate:
                        suppressed_objects.append(track)
                        
                tracked_objects = suppressed_objects
                
                # Tracks active in the current frame
                active_ids = set()
                
                # Split tracked_objects into existing tracks and new tracks
                existing_tracks = []
                new_tracks = []
                for track in tracked_objects:
                    if track.track_id in self.active_tracks:
                        existing_tracks.append(track)
                    else:
                        new_tracks.append(track)

                # Pass 1: Process existing active tracks first to mark them active
                for track in existing_tracks:
                    track_id = track.track_id
                    active_ids.add(track_id)
                    
                    rx1, ry1, rx2, ry2 = map(int, track.tlbr)
                    if track_id not in self.smoothed_bboxes:
                        self.smoothed_bboxes[track_id] = [rx1, ry1, rx2, ry2]
                    else:
                        old_box = self.smoothed_bboxes[track_id]
                        alpha = 0.60
                        self.smoothed_bboxes[track_id] = [
                            int(alpha * rx1 + (1 - alpha) * old_box[0]),
                            int(alpha * ry1 + (1 - alpha) * old_box[1]),
                            int(alpha * rx2 + (1 - alpha) * old_box[2]),
                            int(alpha * ry2 + (1 - alpha) * old_box[3]),
                        ]
                    x1, y1, x2, y2 = self.smoothed_bboxes[track_id]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    
                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size == 0:
                        continue
                        
                    # Face detection
                    y_threshold = int(frame.shape[0] * 0.55)
                    can_have_face = (y1 < y_threshold)
                    faces = self.scrfd.detect(person_crop, simulate=(self.simulate_face and can_have_face))
                    
                    is_valid_face = False
                    if faces:
                        face_info = faces[0]
                        face_score = face_info[4]
                        fx1, fy1, fx2, fy2 = face_info[:4]
                        fw = fx2 - fx1
                        fh = fy2 - fy1
                        
                        # Only accept clear face detections (conf >= 0.55, size >= 24px)
                        if face_score >= 0.55 and fw >= 24 and fh >= 24:
                            is_valid_face = True
                            logger.info(f"Valid face detected for Track {track_id}: score = {face_score:.2f}, size = {fw}x{fh}")
                        else:
                            logger.info(f"Face ignored for Track {track_id}: score = {face_score:.2f}, size = {fw}x{fh} (does not meet criteria)")
                            
                    embedding = None
                    if is_valid_face:
                        face_bbox = faces[0][:4]
                        fx1, fy1, fx2, fy2 = map(int, face_bbox)
                        gx1 = max(0, x1 + fx1)
                        gy1 = max(0, y1 + fy1)
                        gx2 = min(frame.shape[1], x1 + fx2)
                        gy2 = min(frame.shape[0], y1 + fy2)
                        self.track_face_bbox[track_id] = (gx1, gy1, gx2, gy2)
                        
                        # Extract face embedding once
                        face_crop = person_crop[fy1:fy2, fx1:fx2]
                        try:
                            raw_embedding = self.arcface.extract(face_crop)
                            embedding = normalize_embedding(raw_embedding)
                        except Exception as e:
                            logger.warning(f"Failed to extract face embedding for Track {track_id}: {e}")
                    else:
                        self.track_face_bbox.pop(track_id, None)
                        
                    visitor_uuid = self.active_tracks[track_id]
                    
                    # Process face verification and templates
                    if is_valid_face and embedding is not None:
                        if track_id not in self.tracks_with_face:
                            thresh = self.config['pipeline']['faiss_similarity_threshold']
                            matched_uuid, score = self.faiss.search(embedding, threshold=thresh)
                            logger.info(f"Upgrade FAISS search for track {track_id}: score = {score:.4f} (Threshold: {thresh})")
                            
                            old_display_id = self.uuid_to_stable_id.get(visitor_uuid, track_id)
                            if not matched_uuid:
                                new_uuid = uuid.uuid4()
                                self.faiss.add_embedding(embedding, new_uuid)
                                if self.use_db:
                                    self.db.insert_visitor(new_uuid, "face")
                                logger.info(f"Upgraded track {track_id} to new face: {new_uuid}")
                                self.uuid_to_stable_id[new_uuid] = old_display_id
                                visitor_uuid = new_uuid
                            else:
                                logger.info(f"Upgraded track {track_id} to existing face: {matched_uuid}")
                                if matched_uuid not in self.uuid_to_stable_id:
                                    self.uuid_to_stable_id[matched_uuid] = old_display_id
                                visitor_uuid = matched_uuid
                                
                                # Add as additional template if similarity is moderate and template count < 3
                                template_count = self.faiss.uuid_mapping.count(matched_uuid)
                                if template_count < 3 and 0.55 <= score < 0.78:
                                    self.faiss.add_embedding(embedding, matched_uuid)
                                    logger.info(f"Added additional face template ({template_count+1}/3) for visitor UUID: {str(matched_uuid)[:8]} (Similarity score: {score:.2f})")
                                    
                            self.active_tracks[track_id] = visitor_uuid
                            self.tracks_with_face.add(track_id)
                            self.unique_visitors.add(visitor_uuid)
                        else:
                            # Track is already verified, check if we should add another template (e.g. walked closer)
                            template_count = self.faiss.uuid_mapping.count(visitor_uuid)
                            if template_count < 3:
                                _, score = self.faiss.search(embedding, threshold=0.70)
                                if 0.55 <= score < 0.78:
                                    self.faiss.add_embedding(embedding, visitor_uuid)
                                    logger.info(f"Added additional face template ({template_count+1}/3) for visitor UUID: {str(visitor_uuid)[:8]} (Similarity score: {score:.2f})")
                        
                    if visitor_uuid not in self.uuid_to_stable_id:
                        self.uuid_to_stable_id[visitor_uuid] = self.next_display_id
                        self.next_display_id += 1
                        
                    # Update ReID history
                    try:
                        raw_reid = self.reid.extract(person_crop)
                        reid_emb = normalize_embedding(raw_reid)
                        self.reid_history[visitor_uuid] = reid_emb
                    except Exception as e:
                        logger.warning(f"Failed to update ReID history for track {track_id}: {e}")
                        
                    # Mark profile active now
                    self.reid_last_seen[visitor_uuid] = time.time()
                    
                    if self.use_db:
                        self.db.update_live_track(track_id, visitor_uuid)
                        self.db.update_visitor_last_seen(visitor_uuid)

                # Pass 2: Process new tracks second
                for track in new_tracks:
                    track_id = track.track_id
                    active_ids.add(track_id)
                    
                    rx1, ry1, rx2, ry2 = map(int, track.tlbr)
                    if track_id not in self.smoothed_bboxes:
                        self.smoothed_bboxes[track_id] = [rx1, ry1, rx2, ry2]
                    else:
                        old_box = self.smoothed_bboxes[track_id]
                        alpha = 0.60
                        self.smoothed_bboxes[track_id] = [
                            int(alpha * rx1 + (1 - alpha) * old_box[0]),
                            int(alpha * ry1 + (1 - alpha) * old_box[1]),
                            int(alpha * rx2 + (1 - alpha) * old_box[2]),
                            int(alpha * ry2 + (1 - alpha) * old_box[3]),
                        ]
                    x1, y1, x2, y2 = self.smoothed_bboxes[track_id]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    
                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size == 0:
                        continue
                        
                    # Face detection
                    y_threshold = int(frame.shape[0] * 0.55)
                    can_have_face = (y1 < y_threshold)
                    faces = self.scrfd.detect(person_crop, simulate=(self.simulate_face and can_have_face))
                    
                    is_valid_face = False
                    if faces:
                        face_info = faces[0]
                        face_score = face_info[4]
                        fx1, fy1, fx2, fy2 = face_info[:4]
                        fw = fx2 - fx1
                        fh = fy2 - fy1
                        
                        # Only accept clear face detections (conf >= 0.55, size >= 24px)
                        if face_score >= 0.55 and fw >= 24 and fh >= 24:
                            is_valid_face = True
                            logger.info(f"Valid face detected for Track {track_id}: score = {face_score:.2f}, size = {fw}x{fh}")
                        else:
                            logger.info(f"Face ignored for Track {track_id}: score = {face_score:.2f}, size = {fw}x{fh} (does not meet criteria)")
                            
                    if is_valid_face:
                        face_bbox = faces[0][:4]
                        fx1, fy1, fx2, fy2 = map(int, face_bbox)
                        gx1 = max(0, x1 + fx1)
                        gy1 = max(0, y1 + fy1)
                        gx2 = min(frame.shape[1], x1 + fx2)
                        gy2 = min(frame.shape[0], y1 + fy2)
                        self.track_face_bbox[track_id] = (gx1, gy1, gx2, gy2)
                    else:
                        self.track_face_bbox.pop(track_id, None)
                        
                    visitor_uuid = None
                    try:
                        raw_reid = self.reid.extract(person_crop)
                        reid_emb = normalize_embedding(raw_reid)
                        
                        # Get currently active visitor UUIDs in the frame
                        active_uuids = set(self.active_tracks.values())
                        now_time = time.time()
                        
                        best_reid_uuid = None
                        best_reid_score = -1.0
                        for u, cached_emb in self.reid_history.items():
                            if u in active_uuids:
                                continue  # Skip people who are already active in the frame
                            
                            # Skip profiles that were last seen more than 15 seconds ago
                            if now_time - self.reid_last_seen.get(u, 0) > 15.0:
                                continue
                                
                            score = np.dot(reid_emb, cached_emb)
                            if score > best_reid_score:
                                best_reid_score = score
                                best_reid_uuid = u
                                
                        if best_reid_score > 0.70:
                            visitor_uuid = best_reid_uuid
                            logger.info(f"ReID Track Recovery for track {track_id}: matched with UUID {str(visitor_uuid)[:8]} (Score: {best_reid_score:.4f})")
                            self.reid_history[visitor_uuid] = reid_emb  # update cache
                    except Exception as e:
                        logger.warning(f"ReID recovery check failed for track {track_id}: {e}")
                        
                    # If not matched by ReID, process as truly new
                    if not visitor_uuid:
                        self.hourly_total_in += 1  # Truly new visitor entry
                        if is_valid_face:
                            # Extract Face Embedding
                            face_bbox = faces[0][:4]
                            fx1, fy1, fx2, fy2 = map(int, face_bbox)
                            face_crop = person_crop[fy1:fy2, fx1:fx2]
                            face_score = faces[0][4]
                            
                            try:
                                raw_embedding = self.arcface.extract(face_crop)
                                embedding = normalize_embedding(raw_embedding)
                            except Exception as e:
                                logger.warning(f"Failed to extract face embedding: {e}")
                                embedding = None
                                
                            if embedding is not None:
                                self.tracks_with_face.add(track_id)
                                self.track_best_face_score[track_id] = face_score
                                
                                # Search face in FAISS
                                threshold = self.config['pipeline']['faiss_similarity_threshold']
                                matched_uuid, score = self.faiss.search(embedding, threshold=threshold)
                                logger.info(f"New track FAISS search for track {track_id}: score = {score:.4f} (Threshold: {threshold})")
                                
                                if not matched_uuid:
                                    # Register new face permanently
                                    visitor_uuid = uuid.uuid4()
                                    self.faiss.add_embedding(embedding, visitor_uuid)
                                    if self.use_db:
                                        self.db.insert_visitor(visitor_uuid, "face")
                                    logger.info(f"New face visitor registered permanently: {visitor_uuid}")
                                else:
                                    visitor_uuid = matched_uuid
                                    logger.info(f"Existing face visitor recognized: {visitor_uuid} (Score: {score:.2f})")
                                    
                                    # Add as additional template if similarity is moderate and template count < 3
                                    template_count = self.faiss.uuid_mapping.count(matched_uuid)
                                    if template_count < 3 and 0.55 <= score < 0.78:
                                        self.faiss.add_embedding(embedding, matched_uuid)
                                        logger.info(f"Added additional face template ({template_count+1}/3) for visitor UUID: {str(matched_uuid)[:8]} (Similarity score: {score:.2f})")
                                        
                                self.unique_visitors.add(visitor_uuid)
                            else:
                                # Fallback if embedding extraction fails
                                visitor_uuid = uuid.uuid4()
                                if self.use_db:
                                    self.db.insert_visitor(visitor_uuid, "body")
                        else:
                            # Temporary tracking only (no valid face detected)
                            visitor_uuid = uuid.uuid4()
                            if self.use_db:
                                self.db.insert_visitor(visitor_uuid, "body")
                            logger.info(f"Temporary visitor tracked (No Face): {visitor_uuid}")
                            
                        # Save initial ReID embedding for future recoveries
                        try:
                            raw_reid = self.reid.extract(person_crop)
                            reid_emb = normalize_embedding(raw_reid)
                            self.reid_history[visitor_uuid] = reid_emb
                        except Exception as e:
                            logger.warning(f"Failed to extract initial ReID for track {track_id}: {e}")

                    # Register display ID mapping
                    if visitor_uuid not in self.uuid_to_stable_id:
                        self.uuid_to_stable_id[visitor_uuid] = self.next_display_id
                        self.next_display_id += 1

                    self.reid_last_seen[visitor_uuid] = time.time()

                    if self.use_db:
                        self.db.log_event(visitor_uuid, camera_id="imx500")
                        self.db.update_live_track(track_id, visitor_uuid)
                        
                    self.active_tracks[track_id] = visitor_uuid
 
                # Visualization: Draw bounding boxes and IDs
                for track in tracked_objects:
                    track_id = track.track_id
                    x1, y1, x2, y2 = self.smoothed_bboxes.get(track_id, map(int, track.tlbr))
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    
                    visitor_uuid = self.active_tracks.get(track_id, "Unknown")
                    display_id = self.uuid_to_stable_id.get(visitor_uuid, track_id)
                    
                    # State Machine: Green (Verified Face) vs Red (Tracking Only)
                    if track_id in self.tracks_with_face:
                        color = (0, 255, 0) # Green (BGR)
                        label = f"ID: {display_id} | UUID: {str(visitor_uuid)[:8]} | Conf: {track.score:.2f}"
                    else:
                        color = (0, 0, 255) # Red (BGR)
                        label = f"ID: {display_id} | Tracking | Conf: {track.score:.2f}"
                    
                    # Draw body box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Draw face bounding box if detected in this frame
                    if track_id in self.track_face_bbox:
                        fx1, fy1, fx2, fy2 = self.track_face_bbox[track_id]
                        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 255, 0), 2)  # Draw face box in Cyan/Yellow
                        cv2.putText(frame, "Face", (fx1, max(0, fy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
 
                # Premium semi-transparent overlay dashboard at the top-left of the stream
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (320, 95), (0, 0, 0), -1)
                alpha = 0.65  # Transparency factor
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
 
                # Draw the unique visitor metrics text on top of the overlay
                cv2.putText(frame, f"Unique (Session): {len(self.unique_visitors)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(frame, f"Total Registered: {len(self.faiss.uuid_mapping)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                sim_status = "ON" if self.simulate_face else "OFF (Backside)"
                cv2.putText(frame, f"Face Sim (Press 'F'): {sim_status}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
 
                # Encode the frame to JPEG for the HTTP MJPEG stream
                ret_enc, jpeg_buffer = cv2.imencode('.jpg', frame)
                if ret_enc:
                    global latest_frame
                    with frame_lock:
                        latest_frame = jpeg_buffer.tobytes()
 
                # Show the video stream window (safely catch errors if running headlessly)
                try:
                    cv2.imshow("Unique Person Counting", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Quit signal received from video window.")
                        break
                    elif key == ord('f'):
                        self.simulate_face = not self.simulate_face
                        logger.info(f"Toggled face simulation: {self.simulate_face}")
                except Exception:
                    # Sleep slightly if running headlessly to prevent high CPU utilization
                    time.sleep(0.01)
 
                # Calculate exit count before updating active tracks
                exited_ids = set(self.active_tracks.keys()) - active_ids
                self.hourly_total_out += len(exited_ids)

                # Cleanup stale DB and memory tracking states
                self.active_tracks = {tid: uuid for tid, uuid in self.active_tracks.items() if tid in active_ids}
                self.tracks_with_face = {tid for tid in self.tracks_with_face if tid in active_ids}
                self.track_face_bbox = {tid: bbox for tid, bbox in self.track_face_bbox.items() if tid in active_ids}
                self.smoothed_bboxes = {tid: box for tid, box in self.smoothed_bboxes.items() if tid in active_ids}
                self.track_best_face_score = {tid: score for tid, score in self.track_best_face_score.items() if tid in active_ids}
                
                if self.use_db:
                    self.db.delete_stale_tracks()

                # Sync hourly metrics to the database every 5 seconds
                now_time = time.time()
                if now_time - self.last_db_sync_time >= 5.0:
                    self.sync_hourly_metrics()
                    self.last_db_sync_time = now_time

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.camera.stop()

    def sync_hourly_metrics(self):
        if not self.use_db or not self.db:
            return
            
        now = datetime.now()
        # Check if the hour or day has shifted
        if now.hour != self.current_hour or now.date() != self.current_date:
            logger.info(f"Hour shifted from {self.current_hour} to {now.hour}. Resetting hourly metrics.")
            self.hourly_total_in = 0
            self.hourly_total_out = 0
            self.hourly_peak_occupancy = 0
            self.hourly_occupancy_samples = []
            self.current_hour = now.hour
            self.current_date = now.date()
            
        # Compute average occupancy
        if self.hourly_occupancy_samples:
            avg_occ = sum(self.hourly_occupancy_samples) / len(self.hourly_occupancy_samples)
        else:
            avg_occ = 0.0
            
        # Limit ReID history size to keep search fast
        if len(self.reid_history) > 100:
            oldest_keys = list(self.reid_history.keys())[:-100]
            for k in oldest_keys:
                self.reid_history.pop(k, None)
                
        # Upsert metrics to database
        self.db.upsert_people_count_hourly(
            camera_id=self.camera_id,
            total_in=self.hourly_total_in,
            total_out=self.hourly_total_out,
            peak_occupancy=self.hourly_peak_occupancy,
            avg_occupancy=avg_occ
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unique Person Counting System")
    parser.add_argument("--no-db", action="store_true", help="Disable PostgreSQL database logging")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the HTTP MJPEG stream server on")
    args = parser.parse_args()
    
    app = UniquePersonCounter(use_db=not args.no_db, port=args.port)
    app.run()
