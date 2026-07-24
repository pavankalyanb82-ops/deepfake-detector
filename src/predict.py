"""
Phase 7: Image & Video Prediction Inference Module (src/predict.py)
------------------------------------------------------------------
Provides single-image and multi-frame video inference engines with frame aggregation
and Grad-CAM visual heatmaps.

Beginner Explanation:
- How Video Deepfake Detection Works:
  Videos are sequences of individual image frames (e.g. 30 frames per second).
  Rather than analyzing all 300 frames in a 10-second clip (which would be slow),
  we sample 1 frame per second using OpenCV `cv2.VideoCapture`.
- Tradeoff:
  - Higher sampling rate (e.g. 5 frames/sec): Catches brief temporal glitches or face swaps, but runs slower.
  - Lower sampling rate (e.g. 1 frame/sec): Very fast, suitable for real-time web applications!
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.gradcam_explain import GradCAMExplainer

class DeepfakePredictor:
    """
    Main inference interface for Deepfake detection across single images and video clips.
    """
    def __init__(self, checkpoint_path: str = "models/best_model.pth"):
        self.explainer = GradCAMExplainer(checkpoint_path=checkpoint_path)
        
    def predict_image(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
        Runs face extraction, deepfake classification, and Grad-CAM explainability on a single image.
        """
        verdict, prob_fake, confidence, heatmap, cropped_face = self.explainer.explain(pil_image)
        
        return {
            "verdict": verdict,
            "confidence": float(confidence),
            "prob_fake": float(prob_fake),
            "prob_real": float(1.0 - prob_fake),
            "heatmap": heatmap,
            "cropped_face": cropped_face
        }
        
    def predict_video(
        self,
        video_path: str,
        frame_interval_sec: float = 1.0,
        max_frames: int = 15
    ) -> Dict[str, Any]:
        """
        Extracts frames at regular time intervals, runs face detection + scoring per frame,
        and aggregates results into a video-level verdict.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file '{video_path}' not found.")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file '{video_path}' with OpenCV.")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0  # Fallback FPS
            
        frame_step = max(1, int(fps * frame_interval_sec))
        
        frame_results: List[Dict[str, Any]] = []
        frame_count = 0
        sampled_count = 0
        
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break
                
            if frame_count % frame_step == 0:
                # Convert BGR (OpenCV) to RGB (PIL)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                
                # Predict on frame
                res = self.predict_image(pil_frame)
                res["frame_index"] = frame_count
                res["timestamp_sec"] = round(frame_count / fps, 2)
                frame_results.append(res)
                
                sampled_count += 1
                if sampled_count >= max_frames:
                    break
                    
            frame_count += 1
            
        cap.release()
        
        if len(frame_results) == 0:
            raise ValueError("No frames could be extracted from video.")
            
        # Aggregate per-frame probabilities
        prob_fakes = [r["prob_fake"] for r in frame_results]
        avg_prob_fake = float(np.mean(prob_fakes))
        
        video_verdict = "Fake" if avg_prob_fake >= 0.5 else "Real"
        video_confidence = avg_prob_fake if video_verdict == "Fake" else (1.0 - avg_prob_fake)
        
        # Pick the most representative frame (highest prob_fake for Fake verdict, lowest for Real)
        if video_verdict == "Fake":
            rep_frame = max(frame_results, key=lambda x: x["prob_fake"])
        else:
            rep_frame = min(frame_results, key=lambda x: x["prob_fake"])
            
        return {
            "verdict": video_verdict,
            "confidence": float(video_confidence),
            "avg_prob_fake": avg_prob_fake,
            "total_frames_analyzed": len(frame_results),
            "representative_heatmap": rep_frame["heatmap"],
            "representative_face": rep_frame["cropped_face"],
            "representative_timestamp": rep_frame["timestamp_sec"],
            "frame_details": frame_results
        }

def create_sample_video(output_video_path: str = "data/raw/sample_video.mp4", duration_sec: int = 3, fps: int = 30):
    """Utility to generate a small 3-second sample MP4 video for inference testing."""
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (256, 256))
    
    # Load a sample face image from data/raw/fake or real
    sample_img_path = os.path.join("data/raw/fake", os.listdir("data/raw/fake")[0])
    pil_img = Image.open(sample_img_path).resize((256, 256))
    frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    for _ in range(duration_sec * fps):
        writer.write(frame_bgr)
        
    writer.release()
    print(f"[OK] Generated test MP4 video at '{output_video_path}'")

if __name__ == "__main__":
    print("=" * 60)
    print(" PHASE 7: IMAGE & VIDEO INFERENCE TESTING ")
    print("=" * 60)
    
    predictor = DeepfakePredictor()
    
    # 1. Test Image Inference
    sample_img_path = os.path.join("data/raw/fake", os.listdir("data/raw/fake")[0])
    img = Image.open(sample_img_path)
    img_res = predictor.predict_image(img)
    print(f"[OK] Image Inference: Verdict={img_res['verdict']}, Confidence={img_res['confidence']*100:.1f}%")
    
    # 2. Test Video Inference
    test_video_path = "data/raw/sample_video.mp4"
    create_sample_video(test_video_path)
    video_res = predictor.predict_video(test_video_path, frame_interval_sec=1.0)
    print(f"[OK] Video Inference: Verdict={video_res['verdict']}, Confidence={video_res['confidence']*100:.1f}%")
    print(f"     Analyzed {video_res['total_frames_analyzed']} frames across video timeline.")
    print("=" * 60)
