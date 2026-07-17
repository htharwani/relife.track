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
        logger.info("Initializing Centroid Tracker with Track Buffer history")
        self.track_thresh = track_thresh
        self.max_lost_frames = track_buffer
        self.next_id = 1
        # tracks mapping: track_id -> {"centroid": (x, y), "lost_frames": int}
        self.tracks = {}

    def _get_centroid(self, box):
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def update(self, output_results, img_info, img_size):
        online_targets = []
        
        # 1. Gather current detections passing the confidence threshold
        current_centroids = []
        current_boxes = []
        for box in output_results:
            if box[4] > self.track_thresh:
                current_centroids.append(self._get_centroid(box))
                current_boxes.append(box[:4])
                
        # 2. If no current detections, increment lost frames for all tracks and clean up expired ones
        if not current_centroids:
            stale_ids = []
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]["lost_frames"] += 1
                if self.tracks[track_id]["lost_frames"] > self.max_lost_frames:
                    stale_ids.append(track_id)
            for track_id in stale_ids:
                del self.tracks[track_id]
            return online_targets

        # 3. Match current detections to existing/lost tracks
        matched_track_ids = set()
        
        for i, centroid in enumerate(current_centroids):
            best_id = None
            best_dist = float('inf')
            
            for track_id, track_info in self.tracks.items():
                if track_id in matched_track_ids:
                    continue
                prev_centroid = track_info["centroid"]
                dist = np.sqrt((centroid[0] - prev_centroid[0])**2 + (centroid[1] - prev_centroid[1])**2)
                
                # Allow a maximum centroid shift (e.g. 150 pixels) to handle faster movements
                max_dist = 150
                if dist < max_dist and dist < best_dist:
                    best_dist = dist
                    best_id = track_id
                    
            if best_id is not None:
                # Update matched track: reset lost_frames to 0 and update centroid
                self.tracks[best_id] = {"centroid": centroid, "lost_frames": 0}
                matched_track_ids.add(best_id)
                online_targets.append(TrackedObject(best_id, current_boxes[i]))
            else:
                # Create a new track
                self.tracks[self.next_id] = {"centroid": centroid, "lost_frames": 0}
                online_targets.append(TrackedObject(self.next_id, current_boxes[i]))
                self.next_id += 1

        # 4. For all tracks NOT matched in the current frame, increment lost_frames and remove stale ones
        stale_ids = []
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_track_ids:
                self.tracks[track_id]["lost_frames"] += 1
                if self.tracks[track_id]["lost_frames"] > self.max_lost_frames:
                    stale_ids.append(track_id)
        for track_id in stale_ids:
            del self.tracks[track_id]
            
        return online_targets
