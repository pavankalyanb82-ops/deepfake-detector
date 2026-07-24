"""
Phase 1: Environment Verification Script
----------------------------------------
This script checks the Python environment and verifies if PyTorch can access
a GPU (CUDA) or if it will run on CPU.

Note for beginners:
- CUDA is NVIDIA's parallel computing platform that enables GPUs to accelerate deep learning.
- If `torch.cuda.is_available()` returns True, model training and inference will run significantly faster on your GPU.
- If it returns False, PyTorch will automatically default to your CPU. Training works fine on CPU too, just slightly slower!
"""

import sys
import platform

def run_environment_checks():
    print("=" * 60)
    print(" DEEPFAKE DETECTOR - ENVIRONMENT SETUP VERIFICATION ")
    print("=" * 60)
    
    # 1. Check Python version
    python_version = sys.version.split()[0]
    print(f"[OK] Python Version : {python_version} ({platform.system()} {platform.release()})")
    
    # 2. Check PyTorch installation
    try:
        import torch
        import torchvision
        print(f"[OK] PyTorch Version: {torch.__version__}")
        print(f"[OK] torchvision    : {torchvision.__version__}")
        
        # 3. Check GPU / CUDA availability
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            print(f"[OK] CUDA Available : Yes ({device_count} GPU(s) detected)")
            print(f"     Selected GPU  : {gpu_name}")
        else:
            print("[ ! ] CUDA Available : No (PyTorch will use CPU for training and inference)")
            
        print(f"     Default Device : {'cuda' if cuda_available else 'cpu'}")
        
    except ImportError as e:
        print(f"[ X ] PyTorch Error  : Missing PyTorch installation. ({e})")
        
    # 4. Check core CV & Web libraries
    libraries = [
        ("OpenCV", "cv2"),
        ("PIL / Pillow", "PIL"),
        ("FastAPI", "fastapi"),
        ("Streamlit", "streamlit"),
        ("Scikit-Learn", "sklearn"),
        ("Matplotlib", "matplotlib"),
    ]
    
    print("\n--- Core Libraries Status ---")
    for name, module_name in libraries:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "Installed")
            print(f"[OK] {name:<15}: {version}")
        except ImportError:
            print(f"[ ! ] {name:<15}: Not installed (will be installed from requirements.txt)")
            
    print("=" * 60)
    print("Phase 1 verification complete!\n")

if __name__ == "__main__":
    run_environment_checks()
