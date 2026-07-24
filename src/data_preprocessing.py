"""
Phase 3: Data Preprocessing & Augmentation (src/data_preprocessing.py)
----------------------------------------------------------------------
Applies ImageNet normalization, data augmentation, face cropping, and splits data into
Train (70%), Validation (15%), and Test (15%) sets.

Beginner Concepts Explained:
1. Normalization: Standardizes pixel values to match ImageNet mean [0.485, 0.456, 0.406]
   and std [0.229, 0.224, 0.225]. Essential when using pre-trained PyTorch backbones!
2. Data Augmentation: Slightly flips, rotates, or shifts colors of training images.
   This prevents the model from overfitting to specific camera angles or generator quirks.
"""

import os
import sys
import glob
import shutil
from typing import Tuple, Dict
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.face_extraction import FaceExtractor

# Standard ImageNet mean and std values for transfer learning
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(img_size: int = 224) -> Dict[str, transforms.Compose]:
    """
    Returns PyTorch image transformation pipelines for training, validation, and testing.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return {
        'train': train_transform,
        'val': eval_transform,
        'test': eval_transform
    }

class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for Real vs Fake face images.
    Labels: 0 = Real, 1 = Fake
    """
    def __init__(self, file_paths: list, labels: list, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def process_and_split_dataset(raw_dir: str = "data/raw", processed_dir: str = "data/processed") -> None:
    """
    Processes raw images with FaceExtractor and splits dataset into train, val, test folders.
    """
    print("[*] Processing face crops and creating train/val/test splits...")
    extractor = FaceExtractor()
    
    real_paths = glob.glob(os.path.join(raw_dir, "real", "*.*"))
    fake_paths = glob.glob(os.path.join(raw_dir, "fake", "*.*"))
    
    all_paths = real_paths + fake_paths
    all_labels = [0] * len(real_paths) + [1] * len(fake_paths)
    
    if len(all_paths) == 0:
        raise ValueError(f"No images found in {raw_dir}. Run src/download_data.py first.")
        
    # Split 70% train, 30% temp (which gets divided into 15% val, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_paths, all_labels, test_size=0.30, random_state=42, stratify=all_labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    splits = {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }
    
    # Save cropped faces to data/processed/<split>/<class_name>
    for split_name, (paths, labels) in splits.items():
        for path, label in zip(paths, labels):
            class_name = "real" if label == 0 else "fake"
            target_folder = os.path.join(processed_dir, split_name, class_name)
            os.makedirs(target_folder, exist_ok=True)
            
            # Extract face crop
            raw_img = Image.open(path)
            face_crop = extractor.extract_face(raw_img)
            
            file_name = os.path.basename(path)
            save_path = os.path.join(target_folder, file_name)
            face_crop.save(save_path)
            
    print(f"[OK] Preprocessed & split dataset into '{processed_dir}':")
    print(f"     Train: {len(X_train)} samples ({y_train.count(0)} real, {y_train.count(1)} fake)")
    print(f"     Val  : {len(X_val)} samples ({y_val.count(0)} real, {y_val.count(1)} fake)")
    print(f"     Test : {len(X_test)} samples ({y_test.count(0)} real, {y_test.count(1)} fake)")

def get_dataloaders(processed_dir: str = "data/processed", batch_size: int = 16) -> Dict[str, DataLoader]:
    """
    Loads PyTorch DataLoaders for train, val, and test splits.
    """
    transforms_dict = get_transforms()
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        real_files = glob.glob(os.path.join(processed_dir, split, "real", "*.*"))
        fake_files = glob.glob(os.path.join(processed_dir, split, "fake", "*.*"))
        
        paths = real_files + fake_files
        labels = [0] * len(real_files) + [1] * len(fake_files)
        
        dataset = DeepfakeDataset(paths, labels, transform=transforms_dict[split])
        is_shuffle = (split == 'train')
        dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=0)
        
    return dataloaders

if __name__ == "__main__":
    process_and_split_dataset()
    loaders = get_dataloaders()
    for split_name, loader in loaders.items():
        print(f"[OK] {split_name} loader ready: {len(loader.dataset)} items, {len(loader)} batches")
