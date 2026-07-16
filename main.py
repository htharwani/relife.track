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

class UniquePersonCounter:
    def __init__(self, use_db=True):
        self.use_db = use_db
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
        self.detector = HailoYOLODetector(models_cfg['yolo'])
        self.tracker = ByteTrackerWrapper()
        self.scrfd = SCRFDDetector(models_cfg['scrfd'])
        self.arcface = ArcFaceExtractor(models_cfg['arcface'])
        self.reid = RepVGGReID(models_cfg['reid'])
        
        # Init Camera
        self.camera = CameraStream(self.config['camera'])
        
        # State tracking
        self.active_tracks = {} # track_id: visitor_uuid

    def run(self):
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
                    visitor_uuid, score = self.faiss.search(embedding, threshold=self.config['pipeline']['faiss_similarity_threshold'])
                    
                    if not visitor_uuid:
                        # Generate new UUID and store
                        visitor_uuid = uuid.uuid4()
                        self.faiss.add_embedding(embedding, visitor_uuid)
                        if self.use_db:
                            self.db.insert_visitor(visitor_uuid, embedding_type)
                        logger.info(f"New visitor detected: {visitor_uuid} (Type: {embedding_type})")
                    else:
                        logger.info(f"Existing visitor recognized: {visitor_uuid} (Score: {score:.2f})")
                        
                    if self.use_db:
                        self.db.log_event(visitor_uuid, camera_id="imx500")
                        self.db.update_live_track(track_id, visitor_uuid)
                    self.active_tracks[track_id] = visitor_uuid

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
    args = parser.parse_args()
    
    app = UniquePersonCounter(use_db=not args.no_db)
    app.run()
