import cv2
import numpy as np
import face_recognition

class FaceRecognizer:
    def __init__(self):
        # face_recognition handles models internally
        pass
        
    def extract_face(self, frame, bbox):
        """Crop the face from the frame given bounding box [left, top, width, height]."""
        x, y, w, h = bbox
        # Add padding/ensure within frame boundaries
        h_frame, w_frame = frame.shape[:2]
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w_frame, int(x + w)), min(h_frame, int(y + h))
        
        face_img = frame[y1:y2, x1:x2]
        
        if face_img.size == 0:
            return None
        
        # Convert to RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        return face_img
        
    def get_embedding(self, face_img):
        """Generate 128-D embedding from face image."""
        h, w, _ = face_img.shape
        # face_locations = (top, right, bottom, left)
        face_locations = [(0, w, h, 0)] 
        
        embeddings = face_recognition.face_encodings(face_img, known_face_locations=face_locations)
        if len(embeddings) > 0:
            return embeddings[0]
        return None
        
    def match_face(self, embedding, known_users, threshold=0.55):
        """
        Compares the given embedding against known_users [(id, name, embedding)].
        Uses L2 distance.
        Returns (best_match_id, best_match_name, similarity_score) or (None, "Unknown", 0)
        """
        if embedding is None or not known_users:
            return None, "Unknown", 0.0
            
        best_match_id = None
        best_match_name = "Unknown"
        best_distance = float("inf")
        
        for user_id, name, known_emb in known_users:
            # L2 distance
            dist = np.linalg.norm(embedding - known_emb)
            
            if dist < best_distance:
                best_distance = dist
                best_match_id = user_id
                best_match_name = name
                
        # threshold is maximum allowed distance (lower is closer)
        # 0.6 is typical, we use 0.55 to be slightly strict
        if best_distance <= threshold:
            return best_match_id, best_match_name, (1 - best_distance)
        else:
            return None, "Unknown", (1 - best_distance)
