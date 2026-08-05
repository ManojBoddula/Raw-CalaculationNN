---
title: Neural Cal
emoji: 🚀
colorFrom: blue
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: mit
hardware: cpu-basic
short_description: Draw math on a canvas; raw neural networks solve it.
---

# 🧠 Neural Network Calculation Studio

An end-to-end, multi-modal machine learning pipeline that parses, reads, and calculates handwritten mathematical equations directly from a drawing canvas using custom-trained neural networks.

## ✨ Features
- **Custom Computer Vision:** Implements raw OpenCV contour mapping and bounding box extraction to parse handwritten strokes, completely bypassing off-the-shelf OCR tools.
- **Dual Neural Network Architecture:** 
  - **Vision CNN (Stage 1):** Extracts features from handwritten digits and operators (`+`, `-`, `*`) and converts them into tokens.
  - **Math Reasoner NN (Stage 2):** Takes the sequence of neural tokens and predicts the mathematical logic and final result.
- **Explainable AI UI:** A custom-built, glassmorphism web interface that dynamically animates the hidden layer processing of the neural networks step-by-step.
- **Web Speech Integration:** Automatically synthesizes the calculated result using native browser audio APIs.
- **MLOps & CI/CD:** Fully containerized backend deployed via GitHub Actions to Hugging Face Spaces.

## 🛠️ Technology Stack
- **Backend:** FastAPI, Uvicorn, Python
- **Machine Learning:** TensorFlow/Keras, OpenCV, NumPy
- **Frontend:** HTML5 Canvas, Vanilla CSS (Glassmorphism), JavaScript
- **MLOps:** Git LFS, GitHub Actions, Hugging Face Spaces

## 🚀 How it Works
1. **Draw:** The user draws an equation (e.g., `22 + 5`) on the frontend HTML5 Canvas.
2. **Segment:** The image is sent to the FastAPI backend, where custom OpenCV logic segments individual character strokes and handles overlapping bounding boxes.
3. **Vision Processing:** Each isolated character is fed into the Pre-Trained Vision CNN to generate a mathematical token.
4. **Logical Evaluation:** The sequence of tokens is validated and passed to the Math Reasoner NN to compute the final mathematical result.
5. **Animation:** The backend generates dynamic SVGs of the network architecture which the frontend uses to animate the prediction process.

## 💻 Run Locally

### Prerequisites
- Python 3.10+
- Git LFS

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ManojBoddula/Calculationnn-Rawmodel.git
   cd Calculationnn-Rawmodel
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the server:
   ```bash
   python -m uvicorn app:app --host 0.0.0.0 --port 7860
   ```
4. Open your browser and navigate to `http://localhost:7860`.
