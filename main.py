import cv2
import time
import torch
import ultralytics.nn.tasks
from ultralytics import YOLO
from utils.plotting import draw_box, get_color

# Safe handling for PyTorch 2.6 updates
if hasattr(torch.serialization, 'add_safe_globals'):
    try:
        torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
    except Exception:
        pass

def main():
    try:
        print("[INFO] Booting webcam interface...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise IOError("Critical Error: Webcam not found.")
            
        print("[INFO] Loading Generic YOLOv8 Model...")
        # Using the base generic model (handles ~80 categories)
        model = YOLO("yolov8n.pt")
        
        print("[INFO] System running! Press 'Q' to quit gracefully.")
        
        prev_time = time.time()
        
        while True:
            success, frame = cap.read()
            if not success:
                print("[ERROR] Lost connection to camera stream.")
                break
                
            # Resize for consistent engine speed
            frame_resized = cv2.resize(frame, (640, 640))
            
            # Run tracker algorithm to give IDs to objects across frames
            results = model.track(frame_resized, persist=True, tracker="bytetrack.yaml", verbose=False)
            
            class_counts = {}
            
            # Parse detections
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        b = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        class_name = model.names[cls_id] # Generic COCO name (person, dog, apple...)
                        
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        
                        # Get a dynamic color
                        color = get_color(class_name)
                        
                        # Apply to frame
                        draw_box(frame_resized, b, class_name, conf, color)
            
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            # Add Overlay
            y_pos = 30
            cv2.putText(frame_resized, "Stats (Total):", (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            for cls_name, total in class_counts.items():
                y_pos += 30
                text = f"-> {cls_name}: {total}"
                col = get_color(cls_name)
                
                cv2.putText(frame_resized, text, (12, y_pos + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(frame_resized, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                
            cv2.putText(frame_resized, f"FPS: {int(fps)}", (520, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow("OmniVision AI - Generic Tracker", frame_resized)
            
            # Quit on Q
            if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
                print("[INFO] Terminating session...")
                break

    except Exception as e:
        print(f"[CRASH] A critical error occurred: {e}")

    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Stream closed. Goodbye!")

if __name__ == "__main__":
    main()
