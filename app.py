import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
import ultralytics.nn.tasks
from utils.plotting import draw_box, get_color

# Safely handle PyTorch 2.6 loading strictness
if hasattr(torch.serialization, 'add_safe_globals'):
    try:
        torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
    except Exception:
        pass

st.set_page_config(page_title="OmniVision AI", page_icon="👁️", layout="wide")

# ================= SIDEBAR CONFIGURATION =================
st.sidebar.title("⚙️ Model Configuration")
st.sidebar.markdown("Fine-tune the AI settings for optimal results.")

# Valuable settings to adjust model power and strictness!
st.sidebar.markdown("### 1. Model Power")
model_size = st.sidebar.selectbox(
    "Select Model Size", 
    ["Nano (Fastest) - yolov8n", "Small (Balanced) - yolov8s", "Medium (Accurate) - yolov8m"]
)

# Confidence threshold to eliminate noise
st.sidebar.markdown("### 2. Detection Strictness")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.5, 
    step=0.05,
    help="Higher values ignore uncertain objects. Lower values detect more but might be noisy."
)

model_dict = {
    "Nano (Fastest) - yolov8n": "yolov8n.pt",
    "Small (Balanced) - yolov8s": "yolov8s.pt",
    "Medium (Accurate) - yolov8m": "yolov8m.pt"
}
selected_model_path = model_dict[model_size]

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Metrics & Info")
st.sidebar.success("YOLOv8 Generic Model Loaded.\nRecognizing 80+ COCO Categories!")

# ================= CORE LOGIC =================

@st.cache_resource
def load_model(model_path):
    """Cache the model in memory based on the specific version chosen."""
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_frame(frame, model, conf_thresh):
    """Run inference using generic YOLO behavior."""
    frame_resized = cv2.resize(frame, (640, 640))
    results = model.predict(frame_resized, conf=conf_thresh, verbose=False)
    
    class_counts = {}
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                b = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                color = get_color(class_name)
                
                draw_box(frame_resized, b, class_name, conf, color)
                
    return frame_resized, class_counts

# ================= UI LAYOUT =================
def main():
    st.title("👁️ OmniVision AI - Universal Dashboard")
    st.markdown("Advanced Generic Object Detection powered by Deep Learning.")
    
    model = load_model(selected_model_path)
    if not model:
        st.stop()
        
    tab_cam, tab_img = st.tabs(["🎥 Live Camera Feed", "📁 Static Image Upload"])
    
    # TAB: CAMERA
    with tab_cam:
        st.header("Real-Time Analysis")
        col1, col2 = st.columns([1, 1])
        with col1:
            start_btn = st.button("▶️ Start Camera", use_container_width=True)
        with col2:
            stop_btn = st.button("⏹️ Stop Camera", use_container_width=True)
            
        show_raw = st.checkbox("👁️ Show Raw Unprocessed Video", value=False, help="Toggle to see the camera feed without AI overlays.")
        
        frame_placeholder = st.empty()
        
        if "webcam_active" not in st.session_state:
            st.session_state.webcam_active = False

        if start_btn:
            st.session_state.webcam_active = True
        if stop_btn:
            st.session_state.webcam_active = False
            
        if st.session_state.webcam_active:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot connect to webcam.")
                st.session_state.webcam_active = False
            else:
                while st.session_state.webcam_active:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Lost video stream.")
                        break
                        
                    if show_raw:
                        # Just display the raw camera frame
                        raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(raw_rgb, channels="RGB")
                    else:
                        processed_cv, stats = process_frame(frame, model, conf_threshold)
                        processed_rgb = cv2.cvtColor(processed_cv, cv2.COLOR_BGR2RGB)
                        
                        frame_placeholder.image(processed_rgb, channels="RGB")
                    
                cap.release()
                
    # TAB: IMAGE
    with tab_img:
        st.header("Upload Local Image")
        uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            image_pil = Image.open(uploaded_file)
            image_arr = np.array(image_pil)
            
            # Fix RGBA & RGB layout
            if image_arr.shape[-1] == 4:
                image_bgr = cv2.cvtColor(image_arr, cv2.COLOR_RGBA2BGR)
            else:
                image_bgr = cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)
                
            processed_cv, stats = process_frame(image_bgr, model, conf_threshold)
            processed_rgb = cv2.cvtColor(processed_cv, cv2.COLOR_BGR2RGB)
            
            # Using inner tabs so the user can "click" to swap between them
            view_tab1, view_tab2 = st.tabs(["🎯 Processed Output", "📷 Original Image"])
            
            with view_tab1:
                st.image(processed_rgb, use_column_width=True)
            with view_tab2:
                st.image(image_arr, use_column_width=True)
            
            # Display metrics creatively
            if stats:
                st.subheader("Objects Detected:")
                cols = st.columns(4)
                idx = 0
                for obj, count in stats.items():
                    cols[idx % 4].metric(label=obj.capitalize(), value=count)
                    idx += 1
            else:
                st.info("No recognizable objects found at this confidence level.")

if __name__ == "__main__":
    main()
