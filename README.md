# AI-Powered Deepfake Detector

## 🔍 Overview
This AI-powered system detects manipulated images using a deep learning classifier based on Xception. It extracts faces from video/image files, preprocesses them, and predicts if they are real or fake.

## Features
- Video frame extraction and face detection with MTCNN
- Binary classification using Xception
- Trainable and easy-to-use prediction script

## Installation
```bash
pip install tensorflow numpy opencv-python facenet-pytorch matplotlib
```

## Dataset
Place your dataset under `data/train/real/` and `data/train/fake/`.

## Training
```bash
python train.py
```

## Prediction
```bash
python predict.py --image path/to/image.jpg
```
