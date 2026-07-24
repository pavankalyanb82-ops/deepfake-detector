"""
Phase 4: Transfer Learning Model Training (src/train_model.py)
------------------------------------------------------------
Fine-tunes a pre-trained EfficientNet-B0 / ResNet18 model for binary classification (Real vs Fake).

Beginner Concepts Explained:
1. Transfer Learning: Pretrained CNNs (trained on 1.4 million ImageNet photos) already know
   how to detect edges, eye shapes, and textures. We reuse those learned features rather than
   training from scratch.
2. Staged Training:
   - Stage 1: Freeze early backbone layers. Train ONLY the new classifier head for a few epochs.
   - Stage 2: Unfreeze top feature layers and fine-tune with a smaller learning rate.
3. Early Stopping: Halts training if validation loss stops improving to prevent overfitting.
"""

import os
import sys
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import get_dataloaders

class DeepfakeClassifier(nn.Module):
    """
    Transfer learning wrapper for EfficientNet-B0 or ResNet18 backbone.
    Outputs a single binary logit (0 = Real, 1 = Fake).
    """
    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True):
        super(DeepfakeClassifier, self).__init__()
        self.backbone_name = backbone_name
        
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)
            in_features = self.model.fc.in_features
            # Replace final classification head for binary output (1 neuron)
            self.model.fc = nn.Linear(in_features, 1)
        elif backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.model = models.efficientnet_b0(weights=weights)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, 1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose 'resnet18' or 'efficientnet_b0'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def freeze_backbone(self):
        """Freezes all feature extractor layers, training only the final linear head."""
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze final classifier head
        if self.backbone_name == "resnet18":
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif self.backbone_name == "efficientnet_b0":
            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def unfreeze_top_layers(self, unfreeze_blocks: int = 2):
        """Unfreezes top convolutional blocks for fine-tuning."""
        # Enable gradients across all model layers for stage 2 fine-tuning
        for param in self.model.parameters():
            param.requires_grad = True

def train_deepfake_model(
    backbone_name: str = "resnet18",
    epochs_stage1: int = 3,
    epochs_stage2: int = 5,
    batch_size: int = 16,
    lr_stage1: float = 1e-3,
    lr_stage2: float = 1e-4,
    model_save_path: str = "models/best_model.pth"
):
    print("=" * 60)
    print(f" PHASE 4: MODEL TRAINING ({backbone_name.upper()}) ")
    print("=" * 60)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")
    
    # Load dataloaders
    dataloaders = get_dataloaders(batch_size=batch_size)
    
    # Initialize model
    model = DeepfakeClassifier(backbone_name=backbone_name, pretrained=True).to(device)
    
    # Binary Cross Entropy with Logits Loss (handles Sigmoid internally for numerical stability)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())
    
    # ----------------------------------------------------
    # STAGE 1: Train Classifier Head Only (Backbone Frozen)
    # ----------------------------------------------------
    print("\n--- STAGE 1: Training Classifier Head (Backbone Frozen) ---")
    model.freeze_backbone()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_stage1)
    
    for epoch in range(epochs_stage1):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            corrects = 0
            total = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = (torch.sigmoid(outputs) >= 0.5).float()
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                corrects += torch.sum(preds == labels.data).item()
                total += inputs.size(0)
                
            epoch_loss = running_loss / max(total, 1)
            epoch_acc = corrects / max(total, 1)
            print(f"Stage 1 | Epoch {epoch+1}/{epochs_stage1} | {phase.upper():<5} Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.2f}%")
            
            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_weights = copy.deepcopy(model.state_dict())

    # ----------------------------------------------------
    # STAGE 2: Fine-tune Upper Layers (Unfrozen Backbone)
    # ----------------------------------------------------
    print("\n--- STAGE 2: Fine-Tuning Upper Feature Layers ---")
    model.unfreeze_top_layers()
    optimizer = optim.Adam(model.parameters(), lr=lr_stage2)
    patience = 3
    patience_counter = 0
    
    for epoch in range(epochs_stage2):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            corrects = 0
            total = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = (torch.sigmoid(outputs) >= 0.5).float()
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                corrects += torch.sum(preds == labels.data).item()
                total += inputs.size(0)
                
            epoch_loss = running_loss / max(total, 1)
            epoch_acc = corrects / max(total, 1)
            print(f"Stage 2 | Epoch {epoch+1}/{epochs_stage2} | {phase.upper():<5} Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.2f}%")
            
            if phase == 'val':
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_weights = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"[*] Early stopping triggered at epoch {epoch+1}")
                        break
        if patience_counter >= patience:
            break
            
    # Save best model checkpoint
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.load_state_dict(best_model_weights)
    
    checkpoint = {
        'backbone_name': backbone_name,
        'state_dict': model.state_dict(),
        'best_val_loss': best_val_loss
    }
    torch.save(checkpoint, model_save_path)
    print(f"\n[OK] Model training complete! Saved best checkpoint to '{model_save_path}'")
    print("=" * 60)

if __name__ == "__main__":
    train_deepfake_model(backbone_name="resnet18", epochs_stage1=2, epochs_stage2=3)
