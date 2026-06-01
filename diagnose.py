import torch
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

import cv2
import numpy as np
from ultralytics import YOLO
from hsemotion.facial_emotions import HSEmotionRecognizer

# Load models
print("[1] Loading models...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
emotion_model = HSEmotionRecognizer(model_name='enet_b0_8_best_afew', device='cpu')
yolo_model = YOLO("yolov8n.pt")

# Read image
print("[2] Reading image...")
frame = cv2.imread("test_image.png")
if frame is None:
    print("Error: Could not read test_image.png")
    exit(1)
print(f"Image shape: {frame.shape}")

# Convert to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Run Haar Cascade
print("[3] Running Haar Cascade...")
height, width = frame.shape[:2]
min_dim = min(width, height)
min_size = max(24, int(min_dim * 0.03))
faces = face_cascade.detectMultiScale(
    gray, 
    scaleFactor=1.05,
    minNeighbors=3,
    minSize=(min_size, min_size)
)
print(f"Haar Cascade detected {len(faces)} faces.")
for i, (x, y, w, h) in enumerate(faces):
    print(f"  Face {i}: x={x}, y={y}, w={w}, h={h}")

# Run YOLOv8 fallback
print("[4] Running YOLOv8 fallback...")
results = yolo_model.predict(frame, conf=0.4, verbose=False)
fallback_faces = []
for result in results:
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = yolo_model.names[cls_id]
            conf = float(box.conf[0])
            b = box.xyxy[0].cpu().numpy()
            px1, py1, px2, py2 = map(int, b)
            print(f"  YOLO detected: {cls_name} (conf={conf:.2f}), box=[{px1}, {py1}, {px2}, {py2}]")
            
            if cls_id == 0:  # Class 0 is 'person'
                p_width = px2 - px1
                p_height = py2 - py1
                
                # Estimate head bounding box (top 40% of the person's bounding box height)
                hx1 = max(0, px1)
                hy1 = max(0, py1)
                hx2 = min(frame.shape[1], px2)
                hy2 = min(frame.shape[0], int(py1 + p_height * 0.40))
                
                print(f"    Estimated head box: [{hx1}, {hy1}, {hx2}, {hy2}] (w={hx2-hx1}, h={hy2-hy1})")
                if (hx2 - hx1) > 20 and (hy2 - hy1) > 20:
                    fallback_faces.append([hx1, hy1, hx2 - hx1, hy2 - hy1])

# Run HSEmotion if any faces found
test_faces = faces if len(faces) > 0 else np.array(fallback_faces)
print(f"[5] Running HSEmotion on {len(test_faces)} faces...")
for i, (x, y, w, h) in enumerate(test_faces):
    face_roi = frame[y:y+h, x:x+w]
    print(f"  Face ROI shape: {face_roi.shape}")
    if face_roi.size > 0:
        try:
            emotion_en, scores = emotion_model.predict_emotions(face_roi, logits=False)
            print(f"  Predicted: {emotion_en}, confidence: {np.max(scores):.2%}")
        except Exception as e:
            print(f"  Error on face {i}: {e}")
