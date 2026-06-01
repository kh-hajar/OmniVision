import cv2
import zlib

def get_color(class_name: str) -> tuple:
    """
    Generate a stable, distinct BGR color dynamically based on the class name hash.
    This supports infinite classes perfectly.
    """
    crc = zlib.crc32(class_name.encode('utf-8'))
    r = (crc & 0xFF0000) >> 16
    g = (crc & 0x00FF00) >> 8
    b = crc & 0x0000FF
    
    # Increase brightness for dark colors to ensure visibility
    if r + g + b < 300:
        r = min(255, r + 100)
        g = min(255, g + 100)
        b = min(255, b + 100)
        
    return (b, g, r)

def draw_box(frame, box, label: str, confidence: float, color: tuple):
    """
    Draws a bounding box and label with confidence score on the frame.
    """
    x1, y1, x2, y2 = map(int, box)
    
    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Label formatting
    text = f"{label.capitalize()} {confidence * 100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Label background
    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
    
    # Determine text color based on background luminance (YIQ logic)
    luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
    text_color = (0, 0, 0) if luminance > 120 else (255, 255, 255)
    
    # Draw text
    cv2.putText(frame, text, (x1, y1 - 5), font, font_scale, text_color, thickness)
    
    return frame

# Mapping of emotions to fixed BGR colors for clear visual encoding
EMOTION_COLORS = {
    'colère': (0, 0, 255),      # Red
    'mépris': (0, 165, 255),    # Orange
    'dégoût': (0, 100, 80),     # Dark Green/Brown
    'peur': (180, 0, 180),      # Purple
    'joie': (0, 200, 0),        # Green
    'neutre': (180, 180, 180),  # Light Gray
    'tristesse': (255, 0, 0),   # Blue
    'surprise': (0, 215, 255),  # Yellow/Gold
    
    'anger': (0, 0, 255),
    'contempt': (0, 165, 255),
    'disgust': (0, 100, 80),
    'fear': (180, 0, 180),
    'happiness': (0, 200, 0),
    'neutral': (180, 180, 180),
    'sadness': (255, 0, 0),
    'surprise': (0, 215, 255)
}

def get_emotion_color(emotion: str) -> tuple:
    """Return a fixed BGR color for the emotion, defaulting to cyan if unknown."""
    return EMOTION_COLORS.get(emotion.lower(), (255, 255, 0))

