import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch

# ── PyTorch 2.6 compat ────────────────────────────────────────────────────────
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    if not torch.cuda.is_available():
        kwargs["map_location"] = "cpu"
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from ultralytics import YOLO
from hsemotion.facial_emotions import HSEmotionRecognizer
from utils.plotting import draw_box, get_color, get_emotion_color

st.set_page_config(page_title="OmniVision AI", page_icon="👁️",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ══ BASE ══ */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: #f0f4f8 !important;
    color: #0f172a !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ══ SCROLLBAR ══ */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #e2e8f0; }
::-webkit-scrollbar-thumb { background: #2563eb; border-radius: 4px; }

/* ══════════════════════════════════════════
   SIDEBAR — bleu foncé (navy)
══════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.25) !important;
}

/* Tout le texte sidebar en blanc */
[data-testid="stSidebar"],
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #ffffff !important;
}

/* Boutons radio */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 7px !important;
    flex-direction: column !important;
    display: flex !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    padding: 11px 14px !important;
    margin: 0 !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #cbd5e1 !important;
    display: flex !important; align-items: center !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(37,99,235,0.25) !important;
    border-color: rgba(37,99,235,0.6) !important;
    color: #fff !important;
    transform: translateX(3px) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: rgba(37,99,235,0.35) !important;
    border-color: #3b82f6 !important;
    color: #fff !important;
    box-shadow: 0 0 0 1px rgba(59,130,246,0.4) !important;
}

/* Selectbox sidebar */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: #fff !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] svg { fill: #94a3b8 !important; }

/* Slider sidebar */
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #3b82f6 !important;
}

/* ══════════════════════════════════════════
   MAIN CONTENT — blanc clair
══════════════════════════════════════════ */
.main .block-container {
    padding: 0 2.2rem 3rem 2.2rem !important;
    max-width: 1280px !important;
}

/* ── Hero Banner ── */
.hero {
    position: relative; overflow: hidden;
    border-radius: 20px; margin: 1.6rem 0 1.8rem 0;
    padding: 2.2rem 2.8rem;
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 40%, #0ea5e9 100%);
    box-shadow: 0 10px 40px rgba(37,99,235,0.35);
}
.hero::before {
    content: ''; position: absolute; inset: 0; pointer-events: none;
    background:
        radial-gradient(circle at 75% 0%,  rgba(255,255,255,0.14) 0%, transparent 55%),
        radial-gradient(circle at 10% 100%, rgba(255,255,255,0.07) 0%, transparent 45%);
}
.hero-inner { position:relative; z-index:1; display:flex; align-items:center; justify-content:space-between; }
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.35);
    border-radius: 50px; padding: 4px 14px;
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #fff; margin-bottom: 1rem;
}
.hero-dot {
    width: 7px; height: 7px; background: #4ade80;
    border-radius: 50%; animation: blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
.hero-title {
    font-size: 2.5rem !important; font-weight: 900 !important;
    color: #fff !important; line-height: 1.1 !important;
    margin: 0 0 0.6rem 0 !important; letter-spacing: -0.03em;
}
.hero-desc { font-size: 0.95rem; color: rgba(255,255,255,0.85); font-weight: 400; line-height: 1.6; }
.hero-icon {
    font-size: 5rem;
    filter: drop-shadow(0 0 20px rgba(255,255,255,0.25));
    animation: floatY 4s ease-in-out infinite;
}
@keyframes floatY { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }

/* ── Stat cards ── */
.card-row {
    display: grid; grid-template-columns: repeat(3,1fr);
    gap: 1rem; margin-bottom: 1.8rem;
}
.stat-card {
    background: #fff; border-radius: 16px;
    padding: 1.4rem 1.6rem; border: 1px solid #e2e8f0;
    box-shadow: 0 2px 14px rgba(0,0,0,0.06);
    display: flex; align-items: flex-start; gap: 1rem;
    transition: all 0.25s;
}
.stat-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 28px rgba(37,99,235,0.12);
    border-color: #bfdbfe;
}
.stat-icon {
    width: 48px; height: 48px; border-radius: 13px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem; flex-shrink: 0;
}
.ic-blue  { background: #eff6ff; }
.ic-sky   { background: #f0f9ff; }
.ic-teal  { background: #f0fdfa; }
.ic-green { background: #f0fdf4; }
.stat-lbl { font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:#94a3b8; margin-bottom:4px; }
.stat-val { font-size:1.15rem; font-weight:800; color:#0f172a; line-height:1; }
.stat-sub { font-size:0.72rem; color:#64748b; margin-top:3px; }

/* ── Section title ── */
.sec-t {
    font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #64748b;
    margin: 1.4rem 0 0.8rem 0;
    display: flex; align-items: center; gap: 8px;
}
.sec-t::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, #bfdbfe, transparent);
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #fff !important;
    border-radius: 14px 14px 0 0 !important;
    padding: 6px 6px 0 6px !important;
    gap: 4px !important;
    border-bottom: 2px solid #e2e8f0 !important;
    box-shadow: 0 -2px 10px rgba(0,0,0,0.04) !important;
}
button[data-baseweb="tab"] {
    background: transparent !important; border-radius: 10px 10px 0 0 !important;
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    font-size: 0.87rem !important; color: #64748b !important;
    padding: 10px 22px !important; transition: all 0.2s !important; border: none !important;
}
button[data-baseweb="tab"]:hover { color: #2563eb !important; background: #eff6ff !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #1d4ed8 !important; background: #dbeafe !important;
    border-bottom: 3px solid #2563eb !important; font-weight: 700 !important;
}
[data-testid="stTabsContent"] {
    background: #fff !important; border: 1px solid #e2e8f0 !important;
    border-top: none !important; border-radius: 0 0 16px 16px !important;
    padding: 2rem !important; box-shadow: 0 6px 24px rgba(0,0,0,0.06) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #f8faff !important; border: 2px dashed #bfdbfe !important;
    border-radius: 14px !important; padding: 0.8rem !important; transition: all 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
    background: #eff6ff !important; border-color: #2563eb !important;
}
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] small { color: #2563eb !important; font-weight: 500 !important; }
[data-testid="stFileUploadDropzone"] button {
    background: #eff6ff !important; border: 1px solid #bfdbfe !important;
    color: #1d4ed8 !important; border-radius: 8px !important; font-weight: 600 !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Inter', sans-serif !important; font-weight: 700 !important;
    font-size: 0.87rem !important; border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important; transition: all 0.2s !important; border: none !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #0ea5e9) !important;
    color: #fff !important; box-shadow: 0 4px 18px rgba(37,99,235,0.4) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 26px rgba(37,99,235,0.55) !important;
}
.stButton > button:not([kind="primary"]) {
    background: #f1f5f9 !important; color: #475569 !important;
    border: 1.5px solid #e2e8f0 !important;
}
.stButton > button:not([kind="primary"]):hover {
    background: #e2e8f0 !important; color: #0f172a !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #fff !important; border: 1.5px solid #e2e8f0 !important;
    border-radius: 14px !important; padding: 1.1rem 1.3rem !important;
    transition: all 0.22s !important; box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
}
[data-testid="stMetric"]:hover {
    border-color: #bfdbfe !important; transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px rgba(37,99,235,0.12) !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.72rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.09em !important; color: #64748b !important;
}
[data-testid="stMetricValue"] {
    font-size: 2rem !important; font-weight: 900 !important; color: #1d4ed8 !important;
}

/* ── Image ── */
[data-testid="stImage"] img {
    border-radius: 14px !important; box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important;
    max-width: 100% !important;
}

/* ── Banners ── */
.info-bar {
    background: #eff6ff; border: 1px solid #bfdbfe; border-left: 4px solid #2563eb;
    border-radius: 12px; padding: 0.9rem 1.2rem;
    font-size: 0.84rem; color: #1e40af; font-weight: 500;
    margin: 0.8rem 0 1.2rem 0; display: flex; gap: 9px;
}
.warn-bar {
    background: #fffbeb; border: 1px solid #fde68a; border-left: 4px solid #f59e0b;
    border-radius: 12px; padding: 0.9rem 1.2rem;
    font-size: 0.84rem; color: #92400e; font-weight: 500;
    margin: 0.8rem 0 1.2rem 0; display: flex; gap: 9px;
}

/* ── Empty state ── */
.es {
    text-align: center; padding: 3.5rem 1rem;
    border: 2px dashed #bfdbfe; border-radius: 16px;
}
.es .ico { font-size: 3.2rem; opacity:0.4; margin-bottom:0.7rem; }
.es .t   { font-size: 0.95rem; font-weight: 700; color: #475569; }
.es .d   { font-size: 0.8rem; color: #94a3b8; margin-top: 4px; }

/* ── Emotion cards ── */
.emo-card {
    background: #f8faff; border: 1.5px solid #e2e8f0; border-radius: 18px;
    padding: 1.6rem; margin-bottom: 1rem; box-shadow: 0 2px 14px rgba(0,0,0,0.05);
}
.emo-head { display:flex; align-items:center; gap:14px; margin-bottom:1.2rem; }
.emo-ico  { font-size: 2.5rem; line-height: 1; }
.emo-name { font-size: 1.15rem; font-weight: 800; color: #0f172a; }
.emo-conf { font-size: 0.75rem; color: #94a3b8; font-weight: 500; margin-top: 2px; }
.emo-row  { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
.emo-lbl  { font-size:0.75rem; font-weight:600; color:#64748b; min-width:88px; text-align:right; }
.emo-track{ flex:1; height:8px; background:#e2e8f0; border-radius:100px; overflow:hidden; }
.emo-fill { height:100%; border-radius:100px; }
.emo-pct  { font-size:0.72rem; color:#64748b; min-width:36px; font-weight:700; }

/* ── Sidebar logo ── */
.sb-logo { padding:1rem 0 1.5rem 0; text-align:center; }
.sb-logo .sb-ico { font-size:2.8rem; display:block; margin-bottom:0.35rem; }
.sb-logo .sb-name {
    font-size:1.4rem; font-weight:900; letter-spacing:-0.03em; color:#fff !important;
}
.sb-logo .sb-tag {
    font-size:0.62rem; letter-spacing:0.14em; text-transform:uppercase;
    color:rgba(255,255,255,0.45) !important; margin-top:2px;
}
.sb-div { border:none; border-top:1px solid rgba(255,255,255,0.1); margin:0.9rem 0; }
.sb-lbl {
    font-size:0.62rem !important; font-weight:700 !important;
    text-transform:uppercase !important; letter-spacing:0.13em !important;
    color:rgba(255,255,255,0.4) !important; margin-bottom:8px !important; display:block;
}

/* ── Checkbox ── */
[data-testid="stCheckbox"] label { color:#475569 !important; font-size:0.85rem !important; font-weight:500 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
EMOTION_MAP = {
    'anger':'Colère','contempt':'Mépris','disgust':'Dégoût','fear':'Peur',
    'happiness':'Joie','neutral':'Neutre','sadness':'Tristesse','surprise':'Surprise'
}
EMOJI_MAP = {
    'Colère':'😡','Mépris':'😤','Dégoût':'🤢','Peur':'😨',
    'Joie':'😊','Neutre':'😐','Tristesse':'😢','Surprise':'😲'
}
EMO_COLORS = {
    'Colère':'#ef4444','Mépris':'#f97316','Dégoût':'#84cc16','Peur':'#a855f7',
    'Joie':'#22c55e','Neutre':'#94a3b8','Tristesse':'#3b82f6','Surprise':'#f59e0b'
}

# ── Model loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_yolo(path):
    try: return YOLO(path)
    except Exception as e: st.error(f"Erreur YOLO : {e}"); return None

@st.cache_resource
def load_emo_model(name):
    try: return HSEmotionRecognizer(model_name=name, device='cpu')
    except Exception as e: st.error(f"Erreur émotions : {e}"); return None

@st.cache_resource
def load_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

# ── Processing ────────────────────────────────────────────────────────────────
def resize_ar(img, mw=900, mh=900):
    h,w=img.shape[:2]; s=min(mw/w,mh/h)
    return cv2.resize(img,(int(w*s),int(h*s))) if s<1 else img

def run_yolo(frame, model, conf):
    out=frame.copy(); counts={}
    for r in model.predict(frame,conf=conf,verbose=False):
        if r.boxes is None: continue
        for box in r.boxes:
            b=box.xyxy[0].cpu().numpy(); c=float(box.conf[0])
            n=model.names[int(box.cls[0])]; counts[n]=counts.get(n,0)+1
            draw_box(out,b,n,c,get_color(n))
    return resize_ar(out), counts

def run_emotion(frame, cas, emo_model, yolo_fb=None):
    out=frame.copy(); gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ms=max(24,int(min(frame.shape[:2])*0.03))
    faces=cas.detectMultiScale(gray,1.05,3,minSize=(ms,ms))
    if len(faces)==0 and yolo_fb:
        fb=[]
        for r in yolo_fb.predict(frame,conf=0.4,verbose=False):
            if r.boxes is None: continue
            for box in r.boxes:
                if int(box.cls[0])!=0: continue
                b=box.xyxy[0].cpu().numpy(); px1,py1,px2,py2=map(int,b)
                ph=py2-py1; hx2=min(frame.shape[1],px2); hy2=min(frame.shape[0],int(py1+ph*0.4))
                if (hx2-px1)>20 and (hy2-py1)>20: fb.append([px1,py1,hx2-px1,hy2-py1])
        if fb: faces=np.array(fb)
    counts,details={},[]
    for (x,y,w,h) in faces:
        roi=frame[y:y+h,x:x+w]
        if roi.size==0: continue
        try:
            rgb=cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
            en,scores=emo_model.predict_emotions(rgb,logits=False)
            fr=EMOTION_MAP.get(en.lower(),en).capitalize()
            conf=float(np.max(scores)); counts[fr]=counts.get(fr,0)+1
            probs={EMOTION_MAP.get(emo_model.idx_to_class[i].lower(),
                   emo_model.idx_to_class[i]).capitalize():float(p) for i,p in enumerate(scores)}
            details.append({'emotion':fr,'confidence':conf,'probabilities':probs})
            draw_box(out,[x,y,x+w,y+h],fr,conf,get_emotion_color(fr))
        except Exception: pass
    return resize_ar(out), counts, details

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <span class="sb-ico">👁️</span>
        <div class="sb-name">OmniVision</div>
        <div class="sb-tag">AI Vision Platform</div>
    </div>
    <hr class="sb-div">
    """, unsafe_allow_html=True)

    st.markdown('<span class="sb-lbl">🧭 Mode d\'analyse</span>', unsafe_allow_html=True)
    task = st.radio("mode",
        ["🔍 Détection d'objets","🎭 Reconnaissance d'Émotions"],
        label_visibility="collapsed")

    st.markdown('<hr class="sb-div">', unsafe_allow_html=True)
    st.markdown('<span class="sb-lbl">⚙️ Configuration</span>', unsafe_allow_html=True)

    if task == "🔍 Détection d'objets":
        ms = st.selectbox("Modèle YOLOv8",
            ["Nano (Rapide) — yolov8n","Small (Équilibré) — yolov8s","Medium (Précis) — yolov8m"])
        conf_thr = st.slider("Seuil de confiance", 0.10, 1.0, 0.50, 0.05)
        M={"Nano (Rapide) — yolov8n":"yolov8n.pt",
           "Small (Équilibré) — yolov8s":"yolov8s.pt",
           "Medium (Précis) — yolov8m":"yolov8m.pt"}
        sel_model=M[ms]
        st.markdown('<hr class="sb-div">', unsafe_allow_html=True)
        st.markdown("✅ **YOLOv8** — 80+ catégories détectées en temps réel.")
    else:
        el = st.selectbox("Modèle d'émotions",
            ["EfficientNet-B2 (Précis) — enet_b2_8",
             "MobileNet-B0 (Équilibré) — enet_b0_8_best_vgaf",
             "MobileNet-B0 (Rapide) — enet_b0_8_best_afew"])
        EM={"EfficientNet-B2 (Précis) — enet_b2_8":"enet_b2_8",
            "MobileNet-B0 (Équilibré) — enet_b0_8_best_vgaf":"enet_b0_8_best_vgaf",
            "MobileNet-B0 (Rapide) — enet_b0_8_best_afew":"enet_b0_8_best_afew"}
        sel_emo=EM[el]
        st.markdown('<hr class="sb-div">', unsafe_allow_html=True)
        st.markdown("✅ **8 émotions** : Joie, Colère, Tristesse, Surprise, Peur, Mépris, Dégoût, Neutre.")

    st.markdown('<hr class="sb-div">', unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;font-size:0.68rem;color:rgba(255,255,255,0.3);line-height:2;'>"
        "OmniVision AI v2.0<br>YOLOv8 · HSEmotion · OpenCV<br>Module — Image Processing</div>",
        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    is_emo = task == "🎭 Reconnaissance d'Émotions"
    m_icon  = "🎭" if is_emo else "🔍"
    m_mode  = "EMOTION MODE" if is_emo else "DETECTION MODE"
    m_desc  = ("Reconnaissance des émotions faciales via HSEmotion & OpenCV"
               if is_emo else
               "Détection et suivi d'objets en temps réel avec YOLOv8")

    # ── Hero ──
    st.markdown(f"""
    <div class="hero">
      <div class="hero-inner">
        <div>
          <div class="hero-eyebrow"><span class="hero-dot"></span>{m_mode}</div>
          <div class="hero-title">OmniVision AI</div>
          <div class="hero-desc">{m_desc}</div>
        </div>
        <div class="hero-icon">{m_icon}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Stat cards ──
    if is_emo:
        st.markdown("""
        <div class="card-row">
          <div class="stat-card">
            <div class="stat-icon ic-blue">🎭</div>
            <div><div class="stat-lbl">Émotions</div>
                 <div class="stat-val">8 classes</div>
                 <div class="stat-sub">Joie, Colère, Peur…</div></div>
          </div>
          <div class="stat-card">
            <div class="stat-icon ic-sky">👤</div>
            <div><div class="stat-lbl">Détection</div>
                 <div class="stat-val">Multi-visages</div>
                 <div class="stat-sub">Haar Cascade + YOLO</div></div>
          </div>
          <div class="stat-card">
            <div class="stat-icon ic-teal">⚡</div>
            <div><div class="stat-lbl">Modèle</div>
                 <div class="stat-val">HSEmotion</div>
                 <div class="stat-sub">EfficientNet / MobileNet</div></div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card-row">
          <div class="stat-card">
            <div class="stat-icon ic-blue">🧠</div>
            <div><div class="stat-lbl">Catégories</div>
                 <div class="stat-val">80+</div>
                 <div class="stat-sub">Personnes, véhicules…</div></div>
          </div>
          <div class="stat-card">
            <div class="stat-icon ic-sky">🚀</div>
            <div><div class="stat-lbl">Modèle</div>
                 <div class="stat-val">YOLOv8</div>
                 <div class="stat-sub">Nano / Small / Medium</div></div>
          </div>
          <div class="stat-card">
            <div class="stat-icon ic-green">🎯</div>
            <div><div class="stat-lbl">Précision</div>
                 <div class="stat-val">Ajustable</div>
                 <div class="stat-sub">Seuil configurable</div></div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Load models ──
    if not is_emo:
        yolo=load_yolo(sel_model)
        if not yolo: st.stop()
    else:
        emo=load_emo_model(sel_emo); cas=load_cascade(); yolofb=load_yolo("yolov8n.pt")
        if not emo or cas is None or not yolofb: st.stop()

    # ── Tabs ──
    tab1,tab2=st.tabs(["📁  Analyse d'Image","🎥  Flux Caméra en Direct"])

    # ══ TAB IMAGE ══
    with tab1:
        up=st.file_uploader("Glissez-déposez votre image (JPG · PNG · WEBP)",
                             type=["jpg","jpeg","png","webp"])
        if not up:
            st.markdown("""<div class="es">
                <div class="ico">🖼️</div>
                <div class="t">Aucune image sélectionnée</div>
                <div class="d">Importez une image pour lancer l'analyse IA</div>
            </div>""", unsafe_allow_html=True)
        else:
            arr=np.array(Image.open(up))
            bgr=cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR if arr.shape[-1]==4 else cv2.COLOR_RGB2BGR)
            with st.spinner("⚡ Analyse IA en cours…"):
                if not is_emo: pf,stats=run_yolo(bgr,yolo,conf_thr); details=[]
                else:          pf,stats,details=run_emotion(bgr,cas,emo,yolofb)

            pr=cv2.cvtColor(pf,cv2.COLOR_BGR2RGB)
            or_=cv2.cvtColor(resize_ar(bgr),cv2.COLOR_BGR2RGB)

            c1,c2=st.columns(2,gap="large")
            with c1: st.markdown("**🎯 Résultat IA**");    st.image(pr, use_column_width=True)
            with c2: st.markdown("**📷 Image originale**"); st.image(or_,use_column_width=True)

            st.markdown("---")

            if stats:
                lbl="Objets détectés" if not is_emo else "Émotions détectées"
                st.markdown(f'<div class="sec-t">📊 {lbl}</div>',unsafe_allow_html=True)
                sl=sorted(stats.items(),key=lambda x:-x[1])
                cols=st.columns(min(len(sl),5))
                for i,(n,c) in enumerate(sl):
                    em=EMOJI_MAP.get(n,"") if is_emo else ""
                    cols[i%5].metric(f"{em} {n}",c)
            else:
                msg=("Aucun objet détecté. Baissez le seuil de confiance." if not is_emo
                     else "Aucun visage détecté. Assurez-vous que le visage est bien visible.")
                st.markdown(f'<div class="warn-bar">⚠️ {msg}</div>',unsafe_allow_html=True)

            if is_emo and details:
                st.markdown('<div class="sec-t">🧬 Analyse détaillée par visage</div>',unsafe_allow_html=True)
                for i,face in enumerate(details):
                    dom=face["emotion"]; conf=face["confidence"]
                    emoji=EMOJI_MAP.get(dom,"🎭"); color=EMO_COLORS.get(dom,"#2563eb")
                    sp=sorted(face["probabilities"].items(),key=lambda x:-x[1])
                    rows="".join(f"""
                    <div class="emo-row">
                      <div class="emo-lbl">{EMOJI_MAP.get(e,'·')} {e}</div>
                      <div class="emo-track">
                        <div class="emo-fill" style="width:{p*100:.1f}%;background:{EMO_COLORS.get(e,'#2563eb')};"></div>
                      </div>
                      <div class="emo-pct">{p*100:.1f}%</div>
                    </div>""" for e,p in sp)
                    st.markdown(f"""
                    <div class="emo-card">
                      <div class="emo-head">
                        <div class="emo-ico">{emoji}</div>
                        <div>
                          <div class="emo-name" style="color:{color};">Visage #{i+1} — {dom}</div>
                          <div class="emo-conf">Confiance : {conf*100:.1f}%</div>
                        </div>
                      </div>
                      {rows}
                    </div>""", unsafe_allow_html=True)

    # ══ TAB CAMERA ══
    with tab2:
        st.markdown("""<div class="info-bar">
            ℹ️&nbsp; Le mode caméra utilise votre webcam locale. L'analyse est effectuée image par image.
        </div>""", unsafe_allow_html=True)

        ca,cb,cc=st.columns([1,1,3])
        with ca: start=st.button("▶️ Démarrer",type="primary",use_container_width=True)
        with cb: stop =st.button("⏹️ Arrêter", use_container_width=True)
        with cc: raw  =st.checkbox("Flux brut (sans IA)",value=False)

        ph=st.empty(); phs=st.empty()

        if "cam" not in st.session_state: st.session_state.cam=False
        if start: st.session_state.cam=True
        if stop:  st.session_state.cam=False

        if not st.session_state.cam:
            ph.markdown("""<div class="es">
                <div class="ico">📷</div>
                <div class="t">Caméra désactivée</div>
                <div class="d">Cliquez sur ▶️ Démarrer pour activer la webcam</div>
            </div>""", unsafe_allow_html=True)
        else:
            cap=cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Impossible d'accéder à la webcam."); st.session_state.cam=False
            else:
                while st.session_state.cam:
                    ret,frame=cap.read()
                    if not ret: break
                    if raw:
                        ph.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),channels="RGB",use_column_width=True)
                    else:
                        if not is_emo: pf,ps=run_yolo(frame,yolo,conf_thr)
                        else:          pf,ps,_=run_emotion(frame,cas,emo,yolofb)
                        ph.image(cv2.cvtColor(pf,cv2.COLOR_BGR2RGB),channels="RGB",use_column_width=True)
                        if ps:
                            line=" &nbsp;·&nbsp; ".join(
                                f"{EMOJI_MAP.get(k,'') if is_emo else ''} <b style='color:#0f172a'>{k}</b>: {v}"
                                for k,v in ps.items())
                            phs.markdown(
                                f"<div style='font-size:0.82rem;color:#475569;padding:5px 0'>{line}</div>",
                                unsafe_allow_html=True)
                cap.release()


if __name__=="__main__":
    import sys
    from streamlit.web import cli as stcli
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        sys.argv=["streamlit","run",sys.argv[0]]; sys.exit(stcli.main())
    else: main()
