import cv2
import numpy as np

# Colors (BGR format for OpenCV)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)

def draw_bounding_box(frame, bbox, color, text=None):
    """Draws a bounding box and optional text on a frame."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    if text:
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        # Draw background rectangle for text
        cv2.rectangle(frame, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), color, -1)
        # Put text
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
    
    return frame

def resize_frame(frame, width=640):
    """Resizes frame maintaining aspect ratio."""
    h, w = frame.shape[:2]
    ratio = width / w
    new_h = int(h * ratio)
    return cv2.resize(frame, (width, new_h))
