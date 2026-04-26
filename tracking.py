from deep_sort_realtime.deepsort_tracker import DeepSort

class FaceTracker:
    def __init__(self, max_age=30, n_init=3):
        # max_age: Maximum number of missed misses before a track is deleted
        # n_init: Number of consecutive detections before the track is confirmed
        self.tracker = DeepSort(max_age=max_age, n_init=n_init, nms_max_overlap=1.0)
        
    def update(self, detections, frame):
        """
        Takes YOLOv8 detections: [([left, top, w, h], conf, class), ...]
        Returns a list of active tracks.
        """
        # update_tracks expects bounding boxes as [left, top, w, h]
        # and returns a list of Track objects
        tracks = self.tracker.update_tracks(detections, frame=frame)
        return tracks
