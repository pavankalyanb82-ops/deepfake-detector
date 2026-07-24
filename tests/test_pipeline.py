"""
Phase 10: Automated Pytest Test Suite (tests/test_pipeline.py)
--------------------------------------------------------------
Tests face extraction, model inference bounds, and FastAPI REST endpoints.
"""

import os
import sys
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.face_extraction import FaceExtractor
from src.predict import DeepfakePredictor
from api.main import app

@pytest.fixture
def sample_pil_image():
    return Image.new("RGB", (256, 256), color=(220, 180, 150))

@pytest.fixture
def predictor_instance():
    checkpoint_path = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
    return DeepfakePredictor(checkpoint_path=checkpoint_path)

def test_face_extraction(sample_pil_image):
    """Verifies FaceExtractor returns a valid 224x224 RGB cropped image."""
    extractor = FaceExtractor(target_size=(224, 224))
    cropped = extractor.extract_face(sample_pil_image)
    
    assert isinstance(cropped, Image.Image)
    assert cropped.size == (224, 224)
    assert cropped.mode == "RGB"

def test_model_inference_probability(predictor_instance, sample_pil_image):
    """Verifies model prediction returns bounded probability in [0.0, 1.0]."""
    res = predictor_instance.predict_image(sample_pil_image)
    
    assert res["verdict"] in ["Real", "Fake"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert 0.0 <= res["prob_fake"] <= 1.0
    assert 0.0 <= res["prob_real"] <= 1.0
    assert abs((res["prob_fake"] + res["prob_real"]) - 1.0) < 1e-5
    assert isinstance(res["heatmap"], Image.Image)

def test_fastapi_health_endpoint():
    """Verifies FastAPI GET /health returns 200 OK and healthy status."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

def test_fastapi_predict_image_endpoint():
    """Verifies FastAPI POST /predict-image returns 200 OK and expected JSON schema."""
    sample_img_path = os.path.join(PROJECT_ROOT, "data", "raw", "fake", "fake_0000.jpg")
    with open(sample_img_path, "rb") as f:
        img_bytes = f.read()
        
    with TestClient(app) as client:
        response = client.post(
            "/predict-image",
            files={"file": ("fake_0000.jpg", img_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["verdict"] in ["Real", "Fake"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["heatmap_base64"].startswith("data:image/png;base64,")

def test_fastapi_predict_video_endpoint():
    """Verifies FastAPI POST /predict-video returns 200 OK and valid frame analysis JSON."""
    sample_video_path = os.path.join(PROJECT_ROOT, "data", "raw", "sample_video.mp4")
    with open(sample_video_path, "rb") as f:
        vid_bytes = f.read()
        
    with TestClient(app) as client:
        response = client.post(
            "/predict-video",
            files={"file": ("sample_video.mp4", vid_bytes, "video/mp4")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["verdict"] in ["Real", "Fake"]
        assert data["total_frames_analyzed"] > 0
        assert data["heatmap_base64"].startswith("data:image/png;base64,")
