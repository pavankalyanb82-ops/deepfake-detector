# 🛡️ Deepfake Image & Video Detector

An end-to-end Computer Vision & Deep Learning web application designed to detect deepfake images and video clips using **Transfer Learning (ResNet18)**, **Grad-CAM visual explainability**, **FastAPI**, and **Streamlit**.

---

## 📐 Project Architecture & Structure

```
deepfake-detector/
├── data/
│   ├── raw/               # Raw face images & sample MP4 video
│   └── processed/         # Face crops split into train/val/test
├── notebooks/
│   ├── 01_eda.ipynb       # Exploratory Data Analysis Jupyter Notebook
│   └── run_eda.py         # Executable CLI EDA script
├── src/
│   ├── face_extraction.py # MTCNN deep face detection & 224x224 cropping
│   ├── data_preprocessing.py # Augmentation, Normalization, & Dataset splitting
│   ├── train_model.py     # Staged Transfer Learning (ResNet18)
│   ├── evaluate_model.py  # Test set metrics, ROC-AUC curve & confusion matrix
│   ├── gradcam_explain.py # Grad-CAM visual heatmaps (pytorch-grad-cam)
│   ├── predict.py         # Single image & multi-frame video inference
│   └── download_data.py   # Dataset generator & Kaggle download helper
├── models/
│   ├── best_model.pth     # Saved PyTorch checkpoint weights
│   ├── roc_curve.png      # ROC Curve evaluation plot
│   └── sample_gradcam.png # Sample Grad-CAM visual heatmap grid
├── api/
│   ├── main.py            # FastAPI REST backend endpoints
│   └── test_api_server.py # Direct endpoint verification tests
├── app/
│   └── streamlit_app.py   # Streamlit web dashboard frontend
├── tests/
│   └── test_pipeline.py   # Automated pytest unit & integration test suite
├── requirements.txt       # Project dependencies
├── verify_env.py          # Environment verification script
└── README.md              # Project documentation
```

---

## 🧠 Technical Deep-Dive: How it Works

### 1. Transfer Learning (ResNet18 / EfficientNet)
Training deep neural networks from scratch requires massive computing power and millions of labeled faces. **Transfer Learning** allows us to leverage a ResNet18 model pre-trained on ImageNet (1.4 million real-world images). The model already understands fundamental visual features (edges, textures, shapes). We replace the final classification layer with a single binary logit head (`0 = Real`, `1 = Fake`) and fine-tune it in two stages:
- **Stage 1**: Freeze backbone layers and train only the classifier head to align output weights.
- **Stage 2**: Unfreeze upper feature layers with a small learning rate (`1e-4`) to fine-tune representations on subtle facial deepfake artifacts.

### 2. Grad-CAM (Gradient-weighted Class Activation Mapping)
Neural networks are often treated as "black boxes". **Grad-CAM** makes our deepfake detector explainable by calculating the gradients of the model's prediction score with respect to the feature maps in the final convolutional layer (`model.layer4[-1]`). 
- **Red/Yellow regions**: High model attention focus (e.g. eye pupil shape, mouth line, skin blurring).
- **Blue regions**: Low impact areas (e.g. ears, neck, uniform background).

### 3. Face Extraction & Preprocessing
Raw input images often contain distracting background context (walls, clothes, furniture). Feeding uncropped images can cause models to overfit on background noise. We use **MTCNN (Multi-task Cascaded Convolutional Networks)** to detect faces, add a safety margin ratio, and crop the face to a standard `224x224` resolution before feeding it to the model.

---

## 🚀 Quickstart Guide (< 15 Minutes)

### Step 1: Environment Setup
1. Clone the repository and enter directory:
   ```bash
   cd deepfake-detector
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Verify environment setup:
   ```bash
   python verify_env.py
   ```

---

### Step 2: Data Preparation & Preprocessing
1. Populate starter raw dataset (or download Kaggle 140k Real/Fake faces):
   ```bash
   python src/download_data.py
   ```
2. Run face extraction and 70/15/15 train/val/test dataset split:
   ```bash
   python src/data_preprocessing.py
   ```

---

### Step 3: Model Training & Evaluation
1. Train the Transfer Learning model (saves best checkpoint to `models/best_model.pth`):
   ```bash
   python src/train_model.py
   ```
2. Evaluate on the held-out test set (generates metrics & `models/roc_curve.png`):
   ```bash
   python src/evaluate_model.py
   ```
3. Generate sample Grad-CAM heatmap grid (`models/sample_gradcam.png`):
   ```bash
   python src/gradcam_explain.py
   ```

---

### Step 4: Run Application (FastAPI + Streamlit)
1. **Start the FastAPI Backend Server**:
   ```bash
   uvicorn api.main:app --host 127.0.0.1 --port 8000
   ```
   *API documentation will be available live at `http://127.0.0.1:8000/docs`.*

2. **Start the Streamlit Web Dashboard** (in a separate terminal window):
   ```bash
   streamlit run app/streamlit_app.py
   ```
   *Dashboard opens in your browser at `http://localhost:8501`.*

---

### Step 5: Run Automated Tests
Execute the full pytest suite:
```bash
pytest -v tests/test_pipeline.py
```

---

## 📊 Dataset Attribution & Extension

### Primary Dataset: 140k Real and Fake Faces
- **Source**: Kaggle ([140k Real and Fake Faces Dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces))
- **Composition**: Balanced dataset of 70k Real faces from Flickr (FFHQ) and 70k Fake faces generated by StyleGAN.

### Extension Path: FaceForensics++ (Video Deepfakes)
To extend this application to video manipulation benchmarks (Deepfakes, Face2Face, FaceSwap, NeuralTextures):
1. Request access via the official [FaceForensics++ GitHub repository](https://github.com/ondyari/FaceForensics).
2. Download video clips into `data/raw/videos/`.
3. Use `src/predict.py` `predict_video()` function to sample frames at 1-2 FPS.

---

## ⚠️ Limitations & Ethical Considerations

> [!IMPORTANT]
> **Educational Demonstration**: This tool was developed as an educational computer vision project for academic exploration. It is **not** a forensic-grade security tool and should not be relied upon as absolute proof in legal, investigative, or public proceedings.

> [!WARNING]
> **Generator Generalization & Domain Shift**: Deepfake classifiers often learn high-frequency artifacts specific to the generator they were trained on (e.g. StyleGAN). When evaluated against unseen or newer diffusion models (e.g., Midjourney v6, Flux, Sora), detection accuracy can decrease.

> [!CAUTION]
> **Consequences of Errors**:
> - **False Positives** (flagging an authentic face as fake) can cause unwarranted reputational harm to individuals.
> - **False Negatives** (failing to flag a malicious deepfake) can allow deceptive media to spread.
> - Always perform human-in-the-loop verification before drawing conclusions from AI detection scores.
