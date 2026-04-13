# 👁️ OmniVision AI

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green.svg)

**OmniVision AI** is an advanced, universal Real-Time Object Detection application built with cutting-edge Deep Learning techniques. It seamlessly integrates the power of **YOLOv8** within an elegant, highly interactive **Streamlit** dashboard.

## 🚀 Features
- **Universal Recognition**: Natively detects over 80+ object categories in real-time using COCO datasets.
- **Dynamic Power Engine**: Switch on-the-fly between different neural architectures (Nano, Small, Medium) to balance supreme speed and ultimate accuracy.
- **Robust Color Hashing**: Generates unique, distinct, and stable bounding box colors natively using an elegant CRC32 class-hashing algorithm.
- **Interactive UI**: Upload images to get detailed statistics, or use the live camera feed with an option to toggle raw un-processed visuals.

## 📁 Repository Structure
```
OmniVision/
├── main.py               # Standalone OpenCV terminal visualizer
├── app.py                # Graphical Web Interface (Streamlit)
├── requirements.txt      # Automated dependencies list
└── utils/
    └── plotting.py       # Computer Vision drawing and graphing logic
```

## 🛠️ Installation & Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/kh-hajar/OmniVision.git
cd OmniVision
```

2. **Create a secure Virtual Environment and activate it:**
```bash
python -m venv venv
# Windows: .\venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
```

3. **Install the deep-learning dependencies:**
```bash
pip install -r requirements.txt
```

4. **Launch the platform:**
```bash
streamlit run app.py
```
*Note: AI weights (like `yolov8n.pt`) will automatically be downloaded on the first run.*

---
*Built with passion to push the boundaries of accessible computer vision.*
