"""
Phase 5: Model Evaluation Module (src/evaluate_model.py)
-------------------------------------------------------
Evaluates the trained Deepfake Detector model on the held-out test set.

Beginner Concepts Explained:
1. Metrics Breakdown:
   - Accuracy: % of total correct predictions.
   - Precision: When the model predicts 'Fake', how often is it right?
   - Recall: Out of all actual 'Fake' images, how many did the model detect?
   - F1-Score: Balance between Precision and Recall.
   - ROC-AUC: Score from 0 to 1 measuring how well the model separates Real vs Fake across all decision thresholds.

2. Why False Negatives are Dangerous:
   - False Positive (Calling a Real person's face 'Fake'): Causes inconvenience or extra verification step.
   - False Negative (Calling a Malicious Deepfake 'Real'): DANGEROUS! Allows fake media, identity impersonation, or fraud to pass undetected.
   - In production security tools, we often lower the probability decision threshold (e.g. from 0.5 to 0.3) to maximize Recall and flag suspicious content for manual review.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train_model import DeepfakeClassifier
from src.data_preprocessing import get_dataloaders

def evaluate_deepfake_model(
    checkpoint_path: str = "models/best_model.pth",
    save_roc_path: str = "models/roc_curve.png"
):
    print("=" * 60)
    print(" PHASE 5: MODEL EVALUATION ON HELD-OUT TEST SET ")
    print("=" * 60)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found. Run src/train_model.py first.")
        
    # Load model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone_name = checkpoint.get('backbone_name', 'resnet18')
    
    model = DeepfakeClassifier(backbone_name=backbone_name, pretrained=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    # Load test set dataloader
    dataloaders = get_dataloaders(batch_size=16)
    test_loader = dataloaders['test']
    
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            all_probs.extend(probs)
            all_labels.extend(labels.numpy().flatten())
            
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    preds = (all_probs >= 0.5).astype(int)
    
    # Compute performance metrics
    acc = accuracy_score(all_labels, preds)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5
        
    cm = confusion_matrix(all_labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    print("\n--- Performance Metrics ---")
    print(f"Test Accuracy  : {acc * 100:.2f}%")
    print(f"Precision      : {prec * 100:.2f}%")
    print(f"Recall         : {rec * 100:.2f}%")
    print(f"F1-Score       : {f1 * 100:.2f}%")
    print(f"ROC-AUC Score  : {auc:.4f}")
    
    print("\n--- Confusion Matrix ---")
    print(f" True Negatives  (Real -> Real) : {tn}")
    print(f" False Positives (Real -> Fake) : {fp}")
    print(f" False Negatives (Fake -> Real) : {fn}  <-- DANGEROUS ERROR!")
    print(f" True Positives  (Fake -> Fake) : {tp}")
    
    print("\n--- Critical Error Analysis ---")
    print("Why False Negatives are the most dangerous error:")
    print("If a real photo is flagged as fake (False Positive), a human reviewer can quickly verify it.")
    print("However, if a malicious deepfake is flagged as real (False Negative), it bypasses detection.")
    print("To prioritize security, we can lower the threshold from 0.5 to 0.3 to maximize Recall!")
    
    # Plot & Save ROC Curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#3498db', lw=2.5, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#7f8c8d', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    os.makedirs(os.path.dirname(save_roc_path), exist_ok=True)
    plt.savefig(save_roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Saved ROC curve plot to '{save_roc_path}'")
    
    print("\n--- Deepfake Generalization Discussion ---")
    print("Key Lesson in Deepfake Detection:")
    print("Models often learn specific artifacts of the GAN architecture they were trained on (e.g. StyleGAN2).")
    print("When deployed against a different generator (e.g. Midjourney v6, Flux, or FaceSwap), accuracy can drop.")
    print("To generalize well in real-world deployment:")
    print("1. Train on multi-generator datasets (FaceForensics++, DFDC, WildDeepfake).")
    print("2. Apply frequency-domain analysis & pixel artifact augmentation.")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_deepfake_model()
