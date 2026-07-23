import numpy as np
from utils.logger import logger

class TrackedObject:
    def __init__(self, track_id, tlbr, score=0.0):
        self.track_id = track_id
        self.tlbr = tlbr # top-left-bottom-right [x1, y1, x2, y2]
        self.score = score
        self.is_activated = True

class ByteTrackerWrapper:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30):
        logger.info("Initializing Enhanced Tracker with IoU + Centroid matching")
        self.track_thresh = track_thresh
        self.max_lost_frames = track_buffer
        self.next_id = 1
        # tracks mapping: track_id -> {"centroid": (x, y), "box": [x1, y1, x2, y2], "lost_frames": int}
        self.tracks = {}

    def _get_centroid(self, box):
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def _get_iou(self, box1, box2):
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

    def update(self, output_results, img_info, img_size):
        online_targets = []
        
        # 1. Gather current detections passing the confidence threshold
        current_centroids = []
        current_boxes = []
        current_scores = []
        for box in output_results:
            if box[4] > self.track_thresh:
                current_centroids.append(self._get_centroid(box))
                current_boxes.append(box[:4])
                current_scores.append(box[4])
                
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
        matched_det_indices = set()
        matched_track_ids = set()
        
        # Pass 1: Match based on IoU (highly robust for overlapping/close boxes)
        for i, det_box in enumerate(current_boxes):
            best_id = None
            best_iou = 0.3  # Minimum IoU threshold to consider a match
            
            for track_id, track_info in self.tracks.items():
                if track_id in matched_track_ids:
                    continue
                prev_box = track_info.get("box")
                if prev_box is None:
                    continue
                    
                iou = self._get_iou(det_box, prev_box)
                if iou > best_iou:
                    best_iou = iou
                    best_id = track_id
            
            if best_id is not None:
                # Update matched track info
                self.tracks[best_id] = {
                    "centroid": current_centroids[i],
                    "box": det_box,
                    "lost_frames": 0
                }
                matched_track_ids.add(best_id)
                matched_det_indices.add(i)
                online_targets.append(TrackedObject(best_id, det_box, score=current_scores[i]))

        # Pass 2: Match remaining detections using Centroid Distance
        for i, centroid in enumerate(current_centroids):
            if i in matched_det_indices:
                continue
                
            best_id = None
            best_dist = float('inf')
            
            for track_id, track_info in self.tracks.items():
                if track_id in matched_track_ids:
                    continue
                prev_centroid = track_info["centroid"]
                dist = np.sqrt((centroid[0] - prev_centroid[0])**2 + (centroid[1] - prev_centroid[1])**2)
                
                # Increased threshold to 280 pixels to accommodate large shifts and lower frame rates
                max_dist = 280
                if dist < max_dist and dist < best_dist:
                    best_dist = dist
                    best_id = track_id
                    
            if best_id is not None:
                self.tracks[best_id] = {
                    "centroid": centroid,
                    "box": current_boxes[i],
                    "lost_frames": 0
                }
                matched_track_ids.add(best_id)
                matched_det_indices.add(i)
                online_targets.append(TrackedObject(best_id, current_boxes[i], score=current_scores[i]))
            else:
                # Create a new track
                self.tracks[self.next_id] = {
                    "centroid": centroid,
                    "box": current_boxes[i],
                    "lost_frames": 0
                }
                online_targets.append(TrackedObject(self.next_id, current_boxes[i], score=current_scores[i]))
                logger.info(f"Created new Track ID: {self.next_id}")
                self.next_id += 1

        # 4. Increment lost_frames for all unmatched tracks and clear expired ones
        stale_ids = []
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_track_ids:
                self.tracks[track_id]["lost_frames"] += 1
                if self.tracks[track_id]["lost_frames"] > self.max_lost_frames:
                    stale_ids.append(track_id)
        for track_id in stale_ids:
            del self.tracks[track_id]
            
        return online_targets
