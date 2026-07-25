"""
Phase 9: Interactive Web Application (app/streamlit_app.py)
------------------------------------------------------------
Streamlit Frontend Dashboard for Deepfake Image & Video Detection.
Communicates with the FastAPI backend server over HTTP REST APIs with direct fallback.
"""

import os
import sys
import io
import tempfile
import base64
import requests
from PIL import Image
import streamlit as st

# Ensure project root directory is in sys.path for direct imports & deployment
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and modern dark glassmorphism
st.markdown("""
<style>
    /* Global Page Styling */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
        font-family: 'Inter', system-ui, sans-serif;
    }
    
    /* Header Container */
    .header-box {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    /* Verdict Badges */
    .badge-real {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: #ffffff;
        padding: 12px 28px;
        border-radius: 9999px;
        font-size: 26px;
        font-weight: 800;
        display: inline-block;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.5);
    }
    
    .badge-fake {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: #ffffff;
        padding: 12px 28px;
        border-radius: 9999px;
        font-size: 26px;
        font-weight: 800;
        display: inline-block;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.5);
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("🛡️ System Config")
api_base_url = st.sidebar.text_input("FastAPI Server URL", value="http://127.0.0.1:8000")

# Helper function to parse base64 image strings
def base64_to_pil(b64_str: str) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    image_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(image_bytes))

@st.cache_resource
def get_standalone_predictor():
    """Loads predictor directly for standalone Streamlit Cloud deployment or offline mode."""
    from src.predict import DeepfakePredictor
    checkpoint_path = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
    return DeepfakePredictor(checkpoint_path=checkpoint_path)

# Header Banner
st.markdown("""
<div class="header-box">
    <h1 style="margin:0; font-size: 36px; color: #38bdf8;">🛡️ AI Deepfake Image & Video Detector</h1>
    <p style="margin-top: 8px; font-size: 16px; color: #94a3b8;">
        Explainable AI forensic verification using <b>Transfer Learning (ResNet18)</b> and <b>Grad-CAM Visual Attention Maps</b>.
    </p>
</div>
""", unsafe_allow_html=True)

# Navigation Tabs
tab_image, tab_video, tab_info = st.tabs(["📸 Image Detection", "🎥 Video Detection", "ℹ️ System Info & Health"])

# ---------------------------------------------------------
# TAB 1: IMAGE DEEPFAKE DETECTION
# ---------------------------------------------------------
with tab_image:
    st.subheader("Upload Facial Image")
    uploaded_image = st.file_uploader(
        "Choose an image file (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_uploader"
    )
    
    col_input, col_results = st.columns([1, 1])
    
    if uploaded_image is not None:
        with col_input:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
            analyze_btn = st.button("🔍 Analyze Image", type="primary", use_container_width=True)
            
        if analyze_btn:
            with col_results:
                with st.spinner("Analyzing facial features and generating Grad-CAM heatmaps..."):
                    try:
                        # Try FastAPI Backend Endpoint
                        files = {"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
                        response = requests.post(f"{api_base_url}/predict-image", files=files, timeout=5)
                        
                        if response.status_code == 200:
                            data = response.json()
                            verdict = data["verdict"]
                            confidence = data["confidence"]
                            heatmap_b64 = data["heatmap_base64"]
                            face_b64 = data["cropped_face_base64"]
                        else:
                            raise RuntimeError(f"API Error {response.status_code}")
                    except Exception:
                        # Direct Fallback for Streamlit Cloud deployment or offline mode
                        try:
                            predictor = get_standalone_predictor()
                            pil_img = Image.open(uploaded_image).convert("RGB")
                            res = predictor.predict_image(pil_img)
                            
                            verdict = res["verdict"]
                            confidence = res["confidence"]
                            
                            buffer_h = io.BytesIO()
                            res["heatmap"].save(buffer_h, format="PNG")
                            heatmap_b64 = base64.b64encode(buffer_h.getvalue()).decode("utf-8")
                            
                            buffer_f = io.BytesIO()
                            res["cropped_face"].save(buffer_f, format="PNG")
                            face_b64 = base64.b64encode(buffer_f.getvalue()).decode("utf-8")
                        except Exception as fallback_err:
                            st.error(f"Failed to process image: {fallback_err}")
                            st.stop()
                            
                    # Render Verdict Badge
                    if verdict == "Real":
                        st.markdown(f'<div class="badge-real">VERDICT: REAL ({confidence*100:.1f}%)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="badge-fake">VERDICT: FAKE ({confidence*100:.1f}%)</div>', unsafe_allow_html=True)
                        
                    st.write("")
                    st.progress(float(confidence))
                    
                    # Display Side-by-Side Face vs Grad-CAM Heatmap
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(base64_to_pil(face_b64), caption="Extracted Face Crop (224x224)", use_container_width=True)
                    with c2:
                        st.image(base64_to_pil(heatmap_b64), caption="Grad-CAM Attention Focus", use_container_width=True)
                        
                    st.info("💡 **Grad-CAM Interpretation**: Highlighted red/yellow regions show the facial features (pupils, mouth, skin texture) the model focused on to make its prediction.")

# ---------------------------------------------------------
# TAB 2: VIDEO DEEPFAKE DETECTION
# ---------------------------------------------------------
with tab_video:
    st.subheader("Upload Short Video Clip")
    uploaded_video = st.file_uploader(
        "Choose a video file (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader"
    )
    
    col_vid_input, col_vid_results = st.columns([1, 1])
    
    if uploaded_video is not None:
        with col_vid_input:
            st.video(uploaded_video)
            analyze_vid_btn = st.button("🎬 Analyze Video Clip", type="primary", use_container_width=True)
            
        if analyze_vid_btn:
            with col_vid_results:
                with st.spinner("Sampling video frames, analyzing facial geometry, and aggregating timeline..."):
                    try:
                        files = {"file": (uploaded_video.name, uploaded_video.getvalue(), uploaded_video.type)}
                        response = requests.post(f"{api_base_url}/predict-video", files=files, timeout=10)
                        
                        if response.status_code == 200:
                            data = response.json()
                            verdict = data["verdict"]
                            confidence = data["confidence"]
                            total_frames = data["total_frames_analyzed"]
                            rep_timestamp = data["representative_timestamp_sec"]
                            heatmap_b64 = data["heatmap_base64"]
                            face_b64 = data["representative_face_base64"]
                        else:
                            raise RuntimeError(f"API Error {response.status_code}")
                    except Exception:
                        # Direct Fallback for Video
                        try:
                            predictor = get_standalone_predictor()
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                                tmp_file.write(uploaded_video.getvalue())
                                tmp_video_path = tmp_file.name
                                
                            res = predictor.predict_video(tmp_video_path, frame_interval_sec=1.0)
                            if os.path.exists(tmp_video_path):
                                os.remove(tmp_video_path)
                                
                            verdict = res["verdict"]
                            confidence = res["confidence"]
                            total_frames = res["total_frames_analyzed"]
                            rep_timestamp = res["representative_timestamp"]
                            
                            buffer_h = io.BytesIO()
                            res["representative_heatmap"].save(buffer_h, format="PNG")
                            heatmap_b64 = base64.b64encode(buffer_h.getvalue()).decode("utf-8")
                            
                            buffer_f = io.BytesIO()
                            res["representative_face"].save(buffer_f, format="PNG")
                            face_b64 = base64.b64encode(buffer_f.getvalue()).decode("utf-8")
                        except Exception as vid_fallback_err:
                            st.error(f"Failed to process video: {vid_fallback_err}")
                            st.stop()
                            
                    if verdict == "Real":
                        st.markdown(f'<div class="badge-real">VIDEO VERDICT: REAL ({confidence*100:.1f}%)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="badge-fake">VIDEO VERDICT: FAKE ({confidence*100:.1f}%)</div>', unsafe_allow_html=True)
                        
                    st.write("")
                    st.metric("Frames Analyzed", f"{total_frames} sampled frames")
                    st.metric("Key Frame Timestamp", f"{rep_timestamp} seconds")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(base64_to_pil(face_b64), caption=f"Key Frame Face ({rep_timestamp}s)", use_container_width=True)
                    with c2:
                        st.image(base64_to_pil(heatmap_b64), caption="Key Frame Grad-CAM Heatmap", use_container_width=True)

# ---------------------------------------------------------
# TAB 3: SYSTEM INFO & HEALTH
# ---------------------------------------------------------
with tab_info:
    st.subheader("FastAPI Server Health & Diagnostics")
    
    if st.button("🔄 Check Backend API Status"):
        try:
            res = requests.get(f"{api_base_url}/health", timeout=5)
            if res.status_code == 200:
                st.success(f"Backend Server Online: {res.json()}")
            else:
                st.error(f"Health Check Failed: {res.status_code}")
        except Exception as e:
            st.warning(f"Backend Offline ({e}). Running in Standalone Inference Fallback Mode.")
            
    st.markdown("""
    ### Technical Architecture Overview
    - **Backbone Network**: ResNet18 fine-tuned via 2-stage transfer learning.
    - **Face Extraction**: MTCNN deep face detector (crops face to 224x224).
    - **Explainability**: Grad-CAM target layer `model.layer4[-1]`.
    - **Backend**: FastAPI REST server with CORS and error validation.
    """)
