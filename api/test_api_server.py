"""
FastAPI Server Direct Endpoint Test Script
------------------------------------------
"""

import os
import sys
import io
from PIL import Image
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import app

def test_fastapi_endpoints():
    print("=" * 60)
    print(" TESTING FASTAPI BACKEND ENDPOINTS ")
    print("=" * 60)
    
    with TestClient(app) as client:
        # 1. Test /health
        res_health = client.get("/health")
        assert res_health.status_code == 200
        print(f"[OK] GET /health: Status={res_health.status_code}, Response={res_health.json()}")
        
        # 2. Test /predict-image
        sample_img_path = os.path.join(PROJECT_ROOT, "data", "raw", "fake", "fake_0000.jpg")
        with open(sample_img_path, "rb") as f:
            img_bytes = f.read()
            
        res_img = client.post(
            "/predict-image",
            files={"file": ("fake_0000.jpg", img_bytes, "image/jpeg")}
        )
        assert res_img.status_code == 200
        json_img = res_img.json()
        print(f"[OK] POST /predict-image: Verdict={json_img['verdict']}, Confidence={json_img['confidence']*100:.1f}%")
        assert "heatmap_base64" in json_img
        assert json_img["heatmap_base64"].startswith("data:image/png;base64,")
        
        # 3. Test /predict-video
        sample_video_path = os.path.join(PROJECT_ROOT, "data", "raw", "sample_video.mp4")
        with open(sample_video_path, "rb") as f:
            vid_bytes = f.read()
            
        res_vid = client.post(
            "/predict-video",
            files={"file": ("sample_video.mp4", vid_bytes, "video/mp4")}
        )
        assert res_vid.status_code == 200
        json_vid = res_vid.json()
        print(f"[OK] POST /predict-video: Verdict={json_vid['verdict']}, Analyzed {json_vid['total_frames_analyzed']} frames")
        assert "heatmap_base64" in json_vid
        print("=" * 60)

if __name__ == "__main__":
    test_fastapi_endpoints()
