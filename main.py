import cv2
import uuid
from config.config_manager import ConfigManager
from database.postgres_client import PostgresClient
from vector_db.vector_store import VectorStore
from camera.stream import CameraStream
from detector.hailo_yolo import HailoYOLODetector
from tracker.byte_tracker import ByteTrackerWrapper
from face.scrfd import SCRFDDetector
from face.arcface import ArcFaceExtractor
from reid.repvgg import RepVGGReID
from utils.logger import logger
from utils.normalization import normalize_embedding
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time

latest_frame = None
frame_lock = threading.Lock()

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
        
        # Init Models
        models_cfg = self.config['models']
        pipeline_cfg = self.config.get('pipeline', {})
        yolo_thresh = pipeline_cfg.get('yolo_threshold', 0.4)
        face_thresh = pipeline_cfg.get('face_threshold', 0.5)
        
        self.detector = HailoYOLODetector(models_cfg['yolo'], conf_threshold=yolo_thresh)
        self.tracker = ByteTrackerWrapper(track_thresh=yolo_thresh)
        self.scrfd = SCRFDDetector(models_cfg['scrfd'], conf_threshold=face_thresh)
        self.arcface = ArcFaceExtractor(models_cfg['arcface'])
        self.reid = RepVGGReID(models_cfg['reid'])
        
        # Init Camera
        self.camera = CameraStream(self.config['camera'])
        
        # State tracking
        self.active_tracks = {} # track_id: visitor_uuid
        self.unique_visitors = set() # set of unique visitor_uuids seen in this session

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
                boxes = self.detector.detect(frame)
                
                # 2. Tracking
                # Simulated ByteTrack logic mapping
                tracked_objects = self.tracker.update(boxes, frame.shape, frame.shape)
                
                for track in tracked_objects:
                    track_id = track.track_id
                    x1, y1, x2, y2 = map(int, track.tlbr)
                    
                    if track_id in self.active_tracks:
                        # Existing track, just update DB last seen and live tracks
                        visitor_uuid = self.active_tracks[track_id]
                        if self.use_db:
                            self.db.update_live_track(track_id, visitor_uuid)
                            self.db.update_visitor_last_seen(visitor_uuid)
                        self.unique_visitors.add(visitor_uuid)
                        continue
                        
                    # NEW Track detected -> we need to extract embedding
                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size == 0:
                        continue
                        
                    # 3. Face Detection
                    faces = self.scrfd.detect(person_crop)
                    
                    embedding = None
                    embedding_type = "body"
                    
                    if faces:
                        # 4a. Extract Face Embedding
                        face_bbox = faces[0][:4]
                        fx1, fy1, fx2, fy2 = map(int, face_bbox)
                        face_crop = person_crop[fy1:fy2, fx1:fx2]
                        raw_embedding = self.arcface.extract(face_crop)
                        embedding = normalize_embedding(raw_embedding)
                        embedding_type = "face"
                    else:
                        # 4b. Extract Body Embedding (ReID)
                        raw_embedding = self.reid.extract(person_crop)
                        embedding = normalize_embedding(raw_embedding)
                        embedding_type = "body"
                        
                    # 5. Vector Search in FAISS
                    threshold = self.config['pipeline']['faiss_similarity_threshold']
                    # Stricter threshold for mock body ReID color profiles to avoid mismatching similar clothes
                    if embedding_type == "body":
                        threshold = max(threshold, 0.90)
                        
                    visitor_uuid, score = self.faiss.search(embedding, threshold=threshold)
                    
                    if not visitor_uuid:
                        # Generate new UUID and store
                        visitor_uuid = uuid.uuid4()
                        self.faiss.add_embedding(embedding, visitor_uuid)
                        if self.use_db:
                            self.db.insert_visitor(visitor_uuid, embedding_type)
                        logger.info(f"New visitor detected: {visitor_uuid} (Type: {embedding_type})")
                    else:
                        logger.info(f"Existing visitor recognized: {visitor_uuid} (Score: {score:.2f}, Type: {embedding_type})")
                        
                    if self.use_db:
                        self.db.log_event(visitor_uuid, camera_id="imx500")
                        self.db.update_live_track(track_id, visitor_uuid)
                    self.active_tracks[track_id] = visitor_uuid
                    self.unique_visitors.add(visitor_uuid)

                # Visualization: Draw bounding boxes and IDs
                for track in tracked_objects:
                    x1, y1, x2, y2 = map(int, track.tlbr)
                    track_id = track.track_id
                    visitor_uuid = self.active_tracks.get(track_id, "Unknown")
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"ID: {track_id} | UUID: {str(visitor_uuid)[:8]} | Conf: {track.score:.2f}"
                    cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Premium semi-transparent overlay dashboard at the top-left of the stream
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (320, 85), (0, 0, 0), -1)
                alpha = 0.65  # Transparency factor
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # Draw the unique visitor metrics text on top of the overlay
                cv2.putText(frame, f"Unique (Session): {len(self.unique_visitors)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Total Registered: {len(self.faiss.uuid_mapping)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Encode the frame to JPEG for the HTTP MJPEG stream
                ret_enc, jpeg_buffer = cv2.imencode('.jpg', frame)
                if ret_enc:
                    global latest_frame
                    with frame_lock:
                        latest_frame = jpeg_buffer.tobytes()

                # Show the video stream window (safely catch errors if running headlessly)
                try:
                    cv2.imshow("Unique Person Counting", frame)
                    # Exit loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Quit signal received from video window.")
                        break
                except Exception:
                    # Sleep slightly if running headlessly to prevent high CPU utilization
                    time.sleep(0.01)

                # Cleanup stale DB tracks periodically
                if self.use_db:
                    self.db.delete_stale_tracks()

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.camera.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unique Person Counting System")
    parser.add_argument("--no-db", action="store_true", help="Disable PostgreSQL database logging")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the HTTP MJPEG stream server on")
    args = parser.parse_args()
    
    app = UniquePersonCounter(use_db=not args.no_db, port=args.port)
    app.run()
