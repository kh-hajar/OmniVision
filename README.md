# 👁️ OmniVision AI

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green.svg)
![HSEmotion](https://img.shields.io/badge/HSEmotion-FER-purple.svg)

**OmniVision AI** is an advanced, universal AI Vision Platform built with cutting-edge Deep Learning techniques. It combines **real-time object detection** powered by **YOLOv8** and **facial emotion recognition** powered by **HSEmotion**, all wrapped in an elegant, highly interactive **Streamlit** dashboard.

---

## 🚀 Features

### 🔍 Object Detection
- **80+ Categories**: Natively detects over 80 object categories in real-time using the COCO dataset (people, vehicles, animals, everyday objects…).
- **Multi-Model Support**: Switch on-the-fly between three YOLOv8 architectures:
  - **Nano** — ultra-fast inference
  - **Small** — balanced speed/accuracy
  - **Medium** — highest accuracy
- **Configurable Confidence Threshold**: Tune detection sensitivity from 10% to 100%.
- **Robust Color Hashing**: Each class gets a unique, stable bounding box color via a CRC32 hashing algorithm.

### 🎭 Facial Emotion Recognition
- **8 Emotion Classes**: Detects and classifies the following emotions in real-time:
  - 😊 Joy · 😡 Anger · 😢 Sadness · 😲 Surprise
  - 😨 Fear · 😤 Contempt · 🤢 Disgust · 😐 Neutral
- **Dual Face Detection Pipeline**:
  - Primary: **Haar Cascade** (`haarcascade_frontalface_alt2`) for fast detection
  - Fallback: **YOLOv8n** (person class, upper-body crop) for difficult angles and partial visibility
- **Multi-Face Support**: Analyzes all detected faces simultaneously within a single image or video frame.
- **Detailed Per-Face Analysis**: For each face, displays:
  - Dominant emotion with confidence score
  - Full probability distribution across all 8 emotion classes rendered as progress bars
- **3 HSEmotion Model Options**:
  - `EfficientNet-B2` (enet_b2_8) — maximum precision
  - `MobileNet-B0 VGAF` (enet_b0_8_best_vgaf) — balanced
  - `MobileNet-B0 AFEW` (enet_b0_8_best_afew) — fastest inference

### 🖥️ Platform Features
- **Image Analysis Mode**: Upload JPG, PNG, or WEBP images and get instant AI-powered results with side-by-side comparison (original vs. annotated).
- **Live Camera Mode**: Real-time webcam feed with frame-by-frame AI inference and a raw unprocessed view toggle.
- **Premium UI**: Dark sidebar, glassmorphic stat cards, animated hero banner, and smooth micro-animations.

---

## 📁 Repository Structure

```
OmniVision/
├── app.py                # Streamlit Web Interface (main entry point)
├── main.py               # Standalone OpenCV terminal visualizer
├── diagnose.py           # Environment & dependency diagnostic tool
├── requirements.txt      # Python dependencies
├── Guide_Installation.md # Detailed installation guide
└── utils/
    └── plotting.py       # CV drawing utilities (bounding boxes, emotion colors)
```

---

## 🧠 Technology Stack

| Component | Technology |
|---|---|
| Object Detection | [YOLOv8](https://github.com/ultralytics/ultralytics) (Nano / Small / Medium) |
| Emotion Recognition | [HSEmotion](https://github.com/HSE-asavchenko/face-emotion-recognition) (EfficientNet-B2, MobileNet-B0) |
| Face Detection | OpenCV Haar Cascade + YOLOv8n fallback |
| Web Interface | [Streamlit](https://streamlit.io/) 1.32 |
| Computer Vision | [OpenCV](https://opencv.org/) 4.9 |
| Deep Learning Runtime | [PyTorch](https://pytorch.org/) (CPU / GPU) |
| Image Processing | Pillow, NumPy |

---

## 🛠️ Installation & Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/kh-hajar/OmniVision.git
cd OmniVision
```

2. **Create a virtual environment and activate it:**
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3. **Install all dependencies:**
```bash
pip install -r requirements.txt
```

4. **Launch the platform:**
```bash
streamlit run app.py
```

> 💡 **Note**: YOLOv8 weights (`yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`) and HSEmotion model weights will be downloaded automatically on first use.

---

## 🎯 How to Use

### Object Detection
1. Select **🔍 Détection d'objets** in the sidebar.
2. Choose a YOLOv8 model (Nano / Small / Medium).
3. Adjust the confidence threshold slider.
4. Upload an image **or** activate the live camera feed.

### Facial Emotion Recognition
1. Select **🎭 Reconnaissance d'Émotions** in the sidebar.
2. Choose an HSEmotion model (EfficientNet-B2 for best accuracy).
3. Upload an image containing one or more faces **or** activate the live camera feed.
4. View detected faces with bounding boxes colored by dominant emotion and a detailed probability breakdown per face.

---

## 📦 Dependencies

```
ultralytics          # YOLOv8 object detection
opencv-python==4.9.0.80
streamlit==1.32.2
numpy==1.26.4
Pillow==10.2.0
hsemotion            # Facial emotion recognition
timm==0.9.12         # EfficientNet / MobileNet backbone
```

---

*Built with passion to push the boundaries of accessible computer vision.*
