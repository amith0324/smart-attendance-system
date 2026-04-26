import cv2
import time
import argparse
from detection import FaceDetector
from tracking import FaceTracker
from recognition import FaceRecognizer
from database import get_all_users, mark_attendance
from alerts import send_email_alert, send_sms_alert
from utils import draw_bounding_box, resize_frame, COLOR_GREEN, COLOR_RED, COLOR_BLUE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Camera index or video path")
    args = parser.parse_args()

    # Initialize modules
    print("Initializing YOLOv8 Face Detector...")
    detector = FaceDetector()
    
    print("Initializing DeepSORT Tracker...")
    tracker = FaceTracker(max_age=30, n_init=3)
    
    print("Initializing FaceNet Recognizer...")
    recognizer = FaceRecognizer()
    
    print("Loading known users from database...")
    known_users = get_all_users()
    print(f"Loaded {len(known_users)} users.")

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # To keep track of processed IDs to avoid redundant DB calls and alerts
    processed_track_ids = {}
    
    # Unknown faces tracking to send alerts if they persist
    unknown_track_counts = {}

    frame_count = 0
    process_every_n_frames = 2 # Process every 2nd frame for speed (Frame skipping)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frame = resize_frame(frame, width=800)
        
        # We only run detection every N frames to save compute, 
        # but DeepSORT can predict in between if we want. 
        # For simplicity and given YOLOv8n is fast, we run it every frame or skipped frames.
        if frame_count % process_every_n_frames != 0:
            # Optionally just show the frame
            cv2.imshow("Smart Attendance System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
            
        # 1. Detect Faces
        detections = detector.detect(frame, conf_threshold=0.5)
        
        # 2. Track Faces
        tracks = tracker.update(detections, frame)
        
        # 3. Recognize & Process Tracks
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb() # left, top, right, bottom
            x1, y1, x2, y2 = [int(v) for v in ltrb]
            w, h = x2 - x1, y2 - y1
            
            # Default label
            label = f"ID: {track_id} (Unknown)"
            color = COLOR_RED
            
            # Check if we already recognized this track
            if track_id in processed_track_ids:
                user_name = processed_track_ids[track_id]
                if user_name != "Unknown":
                    label = f"{user_name}"
                    color = COLOR_GREEN
            else:
                # Need to recognize
                bbox_for_crop = [x1, y1, w, h]
                face_img = recognizer.extract_face(frame, bbox_for_crop)
                
                if face_img is not None:
                    embedding = recognizer.get_embedding(face_img)
                    user_id, user_name, conf = recognizer.match_face(embedding, known_users, threshold=0.7)
                    
                    if user_id:
                        # Known User
                        processed_track_ids[track_id] = user_name
                        label = f"{user_name}"
                        color = COLOR_GREEN
                        
                        # Mark Attendance
                        success = mark_attendance(user_id)
                        if success:
                            msg = f"Attendance marked for {user_name}."
                            print(msg)
                            send_email_alert("Attendance Marked", msg)
                            # send_sms_alert(msg) # Uncomment to use SMS
                    else:
                        # Unknown User
                        processed_track_ids[track_id] = "Unknown"
                        unknown_track_counts[track_id] = unknown_track_counts.get(track_id, 0) + 1
                        
            # Alert for persistent unknown user
            if processed_track_ids.get(track_id) == "Unknown":
                unknown_track_counts[track_id] = unknown_track_counts.get(track_id, 0) + 1
                if unknown_track_counts[track_id] == 10: # After ~10 frames of being unknown
                    msg = "Alert: Unknown person detected at premises."
                    print(msg)
                    send_email_alert("Security Alert: Unknown Person", msg)
                    
            # Draw bbox and label
            draw_bounding_box(frame, [x1, y1, x2, y2], color, text=label)

        # Show Output
        cv2.imshow("Smart Attendance System", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
