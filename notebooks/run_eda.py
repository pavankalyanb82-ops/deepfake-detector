"""
Phase 2: Executable EDA Script
-----------------------------
Runs Exploratory Data Analysis checks on the raw dataset and prints statistics.
"""

import os
import glob
from PIL import Image

def run_eda(data_dir: str = "data/raw"):
    real_paths = glob.glob(os.path.join(data_dir, "real", "*.*"))
    fake_paths = glob.glob(os.path.join(data_dir, "fake", "*.*"))
    
    print("=" * 60)
    print(" PHASE 2: EXPLORATORY DATA ANALYSIS (EDA) REPORT ")
    print("=" * 60)
    print(f"Dataset Location: '{data_dir}'")
    print(f"Real Images Count: {len(real_paths)}")
    print(f"Fake Images Count: {len(fake_paths)}")
    print(f"Total Images     : {len(real_paths) + len(fake_paths)}")
    
    if len(real_paths) == 0 or len(fake_paths) == 0:
        print("[!] Warning: Missing images in dataset directories.")
        return
        
    sample_real = Image.open(real_paths[0])
    sample_fake = Image.open(fake_paths[0])
    
    print(f"\n[Sample Real Image] Dimensions: {sample_real.size}, Mode: {sample_real.mode}, Format: {sample_real.format}")
    print(f"[Sample Fake Image] Dimensions: {sample_fake.size}, Mode: {sample_fake.mode}, Format: {sample_fake.format}")
    
    balance_ratio = len(real_paths) / max(len(fake_paths), 1)
    print(f"Class Balance Ratio (Real/Fake): {balance_ratio:.2f}")
    if 0.8 <= balance_ratio <= 1.2:
        print("[OK] Dataset is well balanced!")
    else:
        print("[!] Note: Dataset is imbalanced. Consider balancing during training.")
        
    print("\n--- Key GAN Artifacts to Observe in EDA ---")
    print("1. Asymmetric pupils and iris shapes")
    print("2. Irregular jewelry or ear symmetry")
    print("3. Background blur distortions near hair boundaries")
    print("4. Unnatural skin smoothing")
    print("=" * 60)

if __name__ == "__main__":
    run_eda()
