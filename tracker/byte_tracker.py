import numpy as np
from utils.logger import logger

# A mock for the ByteTrack implementation. 
# ByteTrack expects detections in a certain format and returns tracked objects.
# In production, use the actual yolox/tracker/byte_tracker.py or a pip package like `bytetrack`.

class TrackedObject:
    def __init__(self, track_id, tlbr):
        self.track_id = track_id
        self.tlbr = tlbr # top-left-bottom-right [x1, y1, x2, y2]
        self.is_activated = True

class ByteTrackerWrapper:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30):
        logger.info("Initializing Simple Centroid Tracker (ByteTrack fallback)")
        self.track_thresh = track_thresh
        self.next_id = 1
        self.active_tracks = {} # id: centroid (x, y)

    def _get_centroid(self, box):
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def update(self, output_results, img_info, img_size):
        online_targets = []
        if len(output_results) == 0:
            return online_targets
            
        current_centroids = []
        current_boxes = []
        
        for box in output_results:
            if box[4] > self.track_thresh:
                current_centroids.append(self._get_centroid(box))
                current_boxes.append(box[:4])
                
        if not current_centroids:
            return online_targets
            
        new_tracks = {}
        for i, centroid in enumerate(current_centroids):
            best_id = None
            best_dist = float('inf')
            
            for track_id, prev_centroid in self.active_tracks.items():
                dist = np.sqrt((centroid[0] - prev_centroid[0])**2 + (centroid[1] - prev_centroid[1])**2)
                if dist < 100 and dist < best_dist: # 100 pixel max distance
                    best_dist = dist
                    best_id = track_id
                    
            if best_id is not None:
                new_tracks[best_id] = centroid
                online_targets.append(TrackedObject(best_id, current_boxes[i]))
            else:
                new_tracks[self.next_id] = centroid
                online_targets.append(TrackedObject(self.next_id, current_boxes[i]))
                self.next_id += 1
                
        self.active_tracks = new_tracks
        return online_targets
