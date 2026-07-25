"""
Phase 6: Visual Explainability with Grad-CAM (src/gradcam_explain.py)
---------------------------------------------------------------------
Generates Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps that
overlay visual attention maps onto input facial images.

Beginner Concept Explained:
- Deep Learning models are often called 'black boxes'. Grad-CAM unlocks explainability!
  It calculates the gradients of the model's prediction with respect to the final convolutional
  layer.
- Red/Yellow regions on the heatmap show where the model focused its attention (e.g. eye shapes,
  mouth irregularities, background noise) to decide whether a face is REAL or FAKE.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train_model import DeepfakeClassifier
from src.face_extraction import FaceExtractor
from src.data_preprocessing import get_transforms, IMAGENET_MEAN, IMAGENET_STD

class GradCAMExplainer:
    """
    Grad-CAM Visual Explainer for Deepfake Detection models.
    """
    def __init__(self, checkpoint_path: str = "models/best_model.pth"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.backbone_name = checkpoint.get('backbone_name', 'resnet18')
            self.model = DeepfakeClassifier(backbone_name=self.backbone_name, pretrained=False)
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            print(f"[!] Warning: Checkpoint '{checkpoint_path}' not found. Initializing pretrained ResNet18 backbone.")
            self.backbone_name = 'resnet18'
            self.model = DeepfakeClassifier(backbone_name=self.backbone_name, pretrained=True)
            
        self.model.to(self.device)
        self.model.eval()
        
        # Select target layer for Grad-CAM depending on backbone architecture
        if self.backbone_name == "resnet18":
            target_layers = [self.model.model.layer4[-1]]
        elif self.backbone_name == "efficientnet_b0":
            target_layers = [self.model.model.conv_head]
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")
            
        self.cam = GradCAM(model=self.model, target_layers=target_layers)
        self.face_extractor = FaceExtractor(target_size=(224, 224))
        self.eval_transform = get_transforms()['val']

    def explain(self, pil_image: Image.Image) -> tuple:
        """
        Extracts face crop, runs model inference, and generates Grad-CAM heatmap overlay.
        
        Returns:
            - verdict (str): "Fake" or "Real"
            - confidence (float): Probability score in range [0.0, 1.0]
            - heatmap_overlay (PIL.Image.Image): RGB face crop overlaid with Grad-CAM heatmap
            - cropped_face (PIL.Image.Image): Extracted 224x224 input face
        """
        # 1. Extract cropped face
        cropped_face = self.face_extractor.extract_face(pil_image)
        
        # 2. Transform image for PyTorch model
        input_tensor = self.eval_transform(cropped_face).unsqueeze(0).to(self.device)
        
        # 3. Model forward pass
        with torch.no_grad():
            logits = self.model(input_tensor)
            prob_fake = torch.sigmoid(logits).item()
            
        verdict = "Fake" if prob_fake >= 0.5 else "Real"
        confidence = prob_fake if prob_fake >= 0.5 else (1.0 - prob_fake)
        
        # 4. Generate Grad-CAM grayscale activation map
        # Target parameter: None defaults to the highest predicted class
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=None)[0, :]
        
        # 5. Prepare normalized RGB image array [0, 1] for overlaying
        rgb_img_np = np.array(cropped_face).astype(float) / 255.0
        
        # 6. Create heatmap overlay using pytorch-grad-cam utility
        visualization = show_cam_on_image(rgb_img_np, grayscale_cam, use_rgb=True)
        heatmap_overlay = Image.fromarray(visualization)
        
        return verdict, prob_fake, confidence, heatmap_overlay, cropped_face

def generate_sample_explanations(
    raw_data_dir: str = "data/raw",
    output_plot_path: str = "models/sample_gradcam.png"
):
    print("=" * 60)
    print(" PHASE 6: GRAD-CAM VISUAL EXPLAINABILITY ")
    print("=" * 60)
    
    explainer = GradCAMExplainer()
    
    # Pick a real and a fake image sample
    real_sample_path = os.path.join(raw_data_dir, "real", os.listdir(os.path.join(raw_data_dir, "real"))[0])
    fake_sample_path = os.path.join(raw_data_dir, "fake", os.listdir(os.path.join(raw_data_dir, "fake"))[0])
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    for row_idx, (path, label_name) in enumerate([(real_sample_path, "Real"), (fake_sample_path, "Fake")]):
        raw_img = Image.open(path)
        verdict, prob_fake, confidence, heatmap, cropped_face = explainer.explain(raw_img)
        
        # Column 1: Original input face crop
        axes[row_idx, 0].imshow(cropped_face)
        axes[row_idx, 0].set_title(f"Input Face ({label_name})")
        axes[row_idx, 0].axis("off")
        
        # Column 2: Grad-CAM heatmap overlay
        axes[row_idx, 1].imshow(heatmap)
        axes[row_idx, 1].set_title(f"Grad-CAM Heatmap Focus")
        axes[row_idx, 1].axis("off")
        
        # Column 3: Prediction Verdict & Confidence
        color = "#2ecc71" if verdict == "Real" else "#e74c3c"
        axes[row_idx, 2].text(0.1, 0.6, f"Verdict: {verdict}\nProb(Fake): {prob_fake:.1%}\nConf: {confidence:.1%}",
                              fontsize=14, fontweight='bold', color=color, va='center')
        axes[row_idx, 2].axis("off")
        
    plt.suptitle("Deepfake Detector - Grad-CAM Visual Explainability", fontsize=16)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Generated Grad-CAM visualization grid saved to '{output_plot_path}'")
    print("=" * 60)

if __name__ == "__main__":
    generate_sample_explanations()
