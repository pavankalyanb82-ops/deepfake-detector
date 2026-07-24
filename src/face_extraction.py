"""
Phase 3: Face Extraction Module (src/face_extraction.py)
-------------------------------------------------------
Detects and crops human faces from raw input images using MTCNN
(Multi-task Cascaded Convolutional Networks) from `facenet_pytorch`.

Beginner Explanation:
- Why do we crop faces before deepfake detection?
  Raw photos contain complex backgrounds (walls, furniture, clothes). If we feed full images to a CNN,
  the model might overfit by learning background clues rather than actual facial deepfake artifacts.
  Cropping isolates facial regions (eyes, nose, mouth, skin texture).
"""

import os
import sys
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class FaceExtractor:
    """
    Extracts face bounding boxes and crops facial region to 224x224 pixels.
    Uses MTCNN deep learning model for state-of-the-art face detection.
    """
    def __init__(self, target_size=(224, 224), margin_ratio=0.15, device=None):
        self.target_size = target_size
        self.margin_ratio = margin_ratio
        
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Initialize MTCNN face detector
        self.mtcnn = MTCNN(
            image_size=target_size[0],
            margin=20,
            keep_all=False,  # Keep the primary face detected
            post_process=False,
            device=self.device
        )
        
    def extract_face(self, pil_image: Image.Image) -> Image.Image:
        """
        Detects face in PIL image, crops around bounding box, and resizes to target_size.
        If no face is detected, returns a center crop of the image as fallback.
        """
        try:
            # Detect face bounding box using MTCNN
            boxes, probs = self.mtcnn.detect(pil_image)
            
            if boxes is not None and len(boxes) > 0 and probs[0] is not None and probs[0] > 0.6:
                box = boxes[0]  # [x1, y1, x2, y2]
                width, height = pil_image.size
                
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                
                # Add margin ratio
                margin_w = w * self.margin_ratio
                margin_h = h * self.margin_ratio
                
                crop_x1 = max(0, int(x1 - margin_w))
                crop_y1 = max(0, int(y1 - margin_h))
                crop_x2 = min(width, int(x2 + margin_w))
                crop_y2 = min(height, int(y2 + margin_h))
                
                cropped_face = pil_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                return cropped_face.resize(self.target_size, Image.Resampling.BILINEAR)
        except Exception:
            pass
            
        # Fallback: Center crop if MTCNN detector misses face or raises exception
        width, height = pil_image.size
        min_dim = min(width, height)
        start_x = (width - min_dim) // 2
        start_y = (height - min_dim) // 2
        cropped_face = pil_image.crop((start_x, start_y, start_x + min_dim, start_y + min_dim))
        return cropped_face.resize(self.target_size, Image.Resampling.BILINEAR)

if __name__ == "__main__":
    extractor = FaceExtractor()
    test_img = Image.new("RGB", (256, 256), color=(200, 200, 200))
    cropped = extractor.extract_face(test_img)
    print(f"[OK] FaceExtractor (MTCNN) test output size: {cropped.size}")
