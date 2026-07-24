"""
Phase 8: FastAPI REST API Server (api/main.py)
----------------------------------------------
Provides RESTful HTTP endpoints for Deepfake Image and Video detection.

Endpoints:
- `GET /health`         : System health check & model status
- `POST /predict-image` : Upload face image -> returns verdict, confidence, base64 Grad-CAM heatmap
- `POST /predict-video` : Upload video clip -> returns frame aggregated verdict, base64 Grad-CAM heatmap
"""

import os
import sys
import io
import base64
import tempfile
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import DeepfakePredictor

app = FastAPI(
    title="Deepfake Image & Video Detector API",
    description="FastAPI Backend for Transfer Learning & Grad-CAM Deepfake Detection",
    version="1.0.0"
)

# Enable CORS for frontend clients (e.g. Streamlit, React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None

@app.on_event("startup")
def load_model_on_startup():
    global predictor
    checkpoint_path = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"[!] Warning: Checkpoint file '{checkpoint_path}' not found. Ensure model is trained.")
    predictor = DeepfakePredictor(checkpoint_path=checkpoint_path)
    print("[OK] Loaded DeepfakePredictor model into FastAPI backend.")

def pil_to_base64(pil_img: Image.Image) -> str:
    """Converts a PIL Image to a base64-encoded PNG data URI string."""
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

@app.get("/health", tags=["Health Check"])
def health_check():
    """Returns API operational status and model readiness."""
    is_ready = predictor is not None
    return {
        "status": "healthy",
        "model_loaded": is_ready,
        "version": "1.0.0"
    }

@app.post("/predict-image", tags=["Inference"])
async def predict_image_endpoint(file: UploadFile = File(...)):
    """
    Accepts an uploaded face image file (JPEG/PNG/WEBP) and returns Real/Fake prediction,
    confidence score, and base64-encoded Grad-CAM visual heatmap overlay.
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model predictor not initialized.")
        
    # 1. Basic File Validation
    filename = file.filename.lower()
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if not filename.endswith(valid_extensions) and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format '{file.filename}'. Please upload a valid image file ({', '.join(valid_extensions)})."
        )
        
    # 2. Read image bytes
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")
    if len(contents) > 15 * 1024 * 1024:  # 15 MB limit
        raise HTTPException(status_code=400, detail="File size exceeds maximum 15MB limit.")
        
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image file: {str(e)}")
        
    # 3. Run Inference
    res = predictor.predict_image(pil_img)
    
    # 4. Format JSON response with base64 encoded images
    return JSONResponse({
        "success": True,
        "filename": file.filename,
        "verdict": res["verdict"],
        "confidence": res["confidence"],
        "prob_fake": res["prob_fake"],
        "prob_real": res["prob_real"],
        "heatmap_base64": pil_to_base64(res["heatmap"]),
        "cropped_face_base64": pil_to_base64(res["cropped_face"])
    })

@app.post("/predict-video", tags=["Inference"])
async def predict_video_endpoint(file: UploadFile = File(...)):
    """
    Accepts an uploaded video clip (MP4/AVI/MOV/MKV), samples frames, and returns aggregated
    video verdict, confidence score, and representative Grad-CAM heatmap.
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model predictor not initialized.")
        
    filename = file.filename.lower()
    valid_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    if not filename.endswith(valid_extensions) and not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format '{file.filename}'. Please upload a valid video file ({', '.join(valid_extensions)})."
        )
        
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded video file is empty.")
    if len(contents) > 50 * 1024 * 1024:  # 50 MB limit
        raise HTTPException(status_code=400, detail="Video file size exceeds maximum 50MB limit.")
        
    # Save video to temporary file for OpenCV reading
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(contents)
        tmp_video_path = tmp_file.name
        
    try:
        res = predictor.predict_video(tmp_video_path, frame_interval_sec=1.0)
    except Exception as e:
        if os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")
        
    # Clean up temp file
    if os.path.exists(tmp_video_path):
        os.remove(tmp_video_path)
        
    return JSONResponse({
        "success": True,
        "filename": file.filename,
        "verdict": res["verdict"],
        "confidence": res["confidence"],
        "avg_prob_fake": res["avg_prob_fake"],
        "total_frames_analyzed": res["total_frames_analyzed"],
        "representative_timestamp_sec": res["representative_timestamp"],
        "heatmap_base64": pil_to_base64(res["representative_heatmap"]),
        "representative_face_base64": pil_to_base64(res["representative_face"])
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
