import os
import urllib.request
from ultralytics import YOLO
import cv2

MODEL_URL = "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt"
MODEL_PATH = "yolov8n-face.pt"

class FaceDetector:
    def __init__(self):
        self._download_model_if_needed()
        self.model = YOLO(MODEL_PATH)
        
    def _download_model_if_needed(self):
        if not os.path.exists(MODEL_PATH):
            print("Downloading YOLOv8 Face model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Download complete.")

    def detect(self, frame, conf_threshold=0.5):
        """
        Detects faces in a frame using YOLOv8.
        Returns a list of detections formatted for DeepSORT:
        [([left, top, width, height], confidence, detection_class), ...]
        """
        results = self.model(frame, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf >= conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Convert to [left, top, width, height] for DeepSORT
                w = x2 - x1
                h = y2 - y1
                detections.append(([int(x1), int(y1), int(w), int(h)], conf, 'face'))
                
        return detections
