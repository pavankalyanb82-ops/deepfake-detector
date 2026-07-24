"""
Phase 9: Interactive Web Application (app/streamlit_app.py)
------------------------------------------------------------
Streamlit Frontend Dashboard for Deepfake Image & Video Detection.
Communicates with the FastAPI backend server over HTTP REST APIs.
"""

import os
import sys
import io
import base64
import requests
from PIL import Image
import streamlit as st

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
                        # Call FastAPI Backend Endpoint
                        files = {"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
                        response = requests.post(f"{api_base_url}/predict-image", files=files, timeout=30)
                        
                        if response.status_code == 200:
                            data = response.json()
                            verdict = data["verdict"]
                            confidence = data["confidence"]
                            heatmap_b64 = data["heatmap_base64"]
                            face_b64 = data["cropped_face_base64"]
                            
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
                        else:
                            st.error(f"API Error ({response.status_code}): {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Failed to connect to FastAPI backend server at '{api_base_url}'. Error: {e}")
                        st.warning("Please ensure the FastAPI server is running (`uvicorn api.main:app --port 8000`).")

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
                        response = requests.post(f"{api_base_url}/predict-video", files=files, timeout=60)
                        
                        if response.status_code == 200:
                            data = response.json()
                            verdict = data["verdict"]
                            confidence = data["confidence"]
                            total_frames = data["total_frames_analyzed"]
                            rep_timestamp = data["representative_timestamp_sec"]
                            heatmap_b64 = data["heatmap_base64"]
                            face_b64 = data["representative_face_base64"]
                            
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
                        else:
                            st.error(f"API Error ({response.status_code}): {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Failed to connect to FastAPI backend server at '{api_base_url}'. Error: {e}")

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
            st.error(f"Backend Offline or Unreachable: {e}")
            
    st.markdown("""
    ### Technical Architecture Overview
    - **Backbone Network**: ResNet18 fine-tuned via 2-stage transfer learning.
    - **Face Extraction**: MTCNN deep face detector (crops face to 224x224).
    - **Explainability**: Grad-CAM target layer `model.layer4[-1]`.
    - **Backend**: FastAPI REST server with CORS and error validation.
    """)
