# AI-Powered Deepfake Detector

## Overview

The AI-Powered Deepfake Detector is a deep learning-based system designed to identify manipulated facial images, commonly known as deepfakes. It uses the Xception convolutional neural network architecture along with face detection to classify images as real or fake.

This project is ideal for research, digital forensics, and media verification use cases in the data science domain.

## Features

- Face detection from video frames or images using MTCNN.
- Deepfake classification using a binary classifier trained on real and fake images.
- Modular code structure for easy retraining and prediction.
- Lightweight and can be run on consumer hardware with GPU acceleration support.

## Installation

Install the required Python packages using pip:

```bash
pip install tensorflow numpy opencv-python facenet-pytorch matplotlib
```

## Dataset Preparation

1. Organize your dataset in the following structure:

```
data/
└── train/
    ├── real/
    │   ├── image1.jpg
    │   └── ...
    └── fake/
        ├── image2.jpg
        └── ...
```

2. Use your own dataset or download publicly available datasets such as FaceForensics++ or DeepFake Detection Challenge datasets.

## Usage

### 1. Extract Faces from Video

```bash
python extract_faces.py
```

Edit `extract_faces.py` to provide your video path and desired save directory.

### 2. Train the Model

```bash
python train.py
```

Model will be saved as `deepfake_detector_model.h5`.

### 3. Make Predictions

```bash
python predict.py --image path/to/image.jpg
```

Output will display whether the image is real or fake along with prediction confidence.

## Model

- Base Model: Xception (pre-trained on ImageNet)
- Custom head: GlobalAveragePooling + Dense layers
- Loss: Binary Crossentropy
- Optimizer: Adam

## Project Structure

```
deepfake_detector/
├── extract_faces.py
├── train.py
├── predict.py
└── README.md
```
