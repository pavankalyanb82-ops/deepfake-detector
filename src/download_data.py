"""
Data Downloader & Sample Dataset Generator
------------------------------------------
This script generates or downloads a starter sample dataset into `data/raw/real` and `data/raw/fake`.

For full training, download the '140k Real and Fake Faces' dataset from Kaggle:
https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
and place the images inside:
  - `data/raw/real/`
  - `data/raw/fake/`

This script populates synthetic sample images for initial testing so the pipeline works out-of-the-box.
"""

import os
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def generate_sample_dataset(num_samples_per_class: int = 50, data_dir: str = "data/raw"):
    """
    Generates a starter set of clean 'real' lookalike face images and 'fake' images
    with intentional GAN-like artifacts (asymmetric patterns, frequency noise, unnatural blur).
    """
    real_dir = os.path.join(data_dir, "real")
    fake_dir = os.path.join(data_dir, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    print(f"[*] Generating sample dataset in '{data_dir}' ({num_samples_per_class} per class)...")
    
    np.random.seed(42)

    for i in range(num_samples_per_class):
        # 1. Create base face-like oval structure (256x256)
        img_size = (256, 256)
        
        # --- Generate Real Sample (smooth facial gradient, symmetric features) ---
        real_img = Image.new("RGB", img_size, color=(240, 210, 190))
        draw_real = ImageDraw.Draw(real_img)
        # Face skin tone background gradient
        # Eyes (Symmetric)
        draw_real.ellipse([70, 90, 105, 125], fill=(255, 255, 255), outline=(50, 30, 20), width=2)
        draw_real.ellipse([150, 90, 185, 125], fill=(255, 255, 255), outline=(50, 30, 20), width=2)
        draw_real.ellipse([82, 102, 93, 113], fill=(40, 70, 120))  # Left pupil
        draw_real.ellipse([162, 102, 173, 113], fill=(40, 70, 120)) # Right pupil
        # Nose
        draw_real.line([(128, 115), (122, 155), (134, 155)], fill=(180, 130, 100), width=3)
        # Mouth
        draw_real.arc([95, 170, 161, 200], start=0, end=180, fill=(200, 60, 60), width=4)
        
        real_path = os.path.join(real_dir, f"real_{i:04d}.jpg")
        real_img.save(real_path, quality=95)

        # --- Generate Fake Sample (with typical GAN artifacts: pupil asymmetry, background noise, skin warping) ---
        fake_img = Image.new("RGB", img_size, color=(235, 205, 185))
        draw_fake = ImageDraw.Draw(fake_img)
        # Eyes (Asymmetric - common StyleGAN artifact!)
        draw_fake.ellipse([68, 88, 108, 128], fill=(255, 255, 255), outline=(50, 30, 20), width=2)
        draw_fake.ellipse([152, 92, 182, 122], fill=(250, 240, 240), outline=(50, 30, 20), width=3) # asymmetric size/shape
        draw_fake.ellipse([80, 100, 96, 116], fill=(30, 30, 30))  # Irregular pupil shape
        draw_fake.ellipse([163, 103, 171, 111], fill=(80, 40, 40)) # Misaligned pupil color/size
        # Nose
        draw_fake.line([(126, 115), (120, 153), (137, 156)], fill=(170, 120, 90), width=4)
        # Mouth
        draw_fake.arc([90, 168, 165, 205], start=10, end=170, fill=(210, 50, 70), width=4)
        # GAN Artifact: Random high-frequency noise / background warping
        arr_fake = np.array(fake_img)
        noise = np.random.normal(0, 15, arr_fake.shape).astype(np.uint8)
        arr_fake = np.clip(arr_fake.astype(int) + noise, 0, 255).astype(np.uint8)
        fake_img_corrupted = Image.fromarray(arr_fake)
        
        fake_path = os.path.join(fake_dir, f"fake_{i:04d}.jpg")
        fake_img_corrupted.save(fake_path, quality=95)

    print(f"[OK] Created {num_samples_per_class} real samples in '{real_dir}'")
    print(f"[OK] Created {num_samples_per_class} fake samples in '{fake_dir}'")

def check_dataset_status(data_dir: str = "data/raw"):
    real_images = glob.glob(os.path.join(data_dir, "real", "*.[jJ][pP][gG]")) + \
                  glob.glob(os.path.join(data_dir, "real", "*.[pP][nN][gG]"))
    fake_images = glob.glob(os.path.join(data_dir, "fake", "*.[jJ][pP][gG]")) + \
                  glob.glob(os.path.join(data_dir, "fake", "*.[pP][nN][gG]"))
    
    print(f"Dataset Status ('{data_dir}'):")
    print(f"  Real images found: {len(real_images)}")
    print(f"  Fake images found: {len(fake_images)}")
    
    return len(real_images), len(fake_images)

if __name__ == "__main__":
    real_cnt, fake_cnt = check_dataset_status()
    if real_cnt == 0 or fake_cnt == 0:
        generate_sample_dataset(num_samples_per_class=50)
