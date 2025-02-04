# Video Classification with TabNet and CLIP

![GitHub](https://img.shields.io/github/license/ShockOfWave/bubbles_champagne)
![GitHub last commit](https://img.shields.io/github/last-commit/ShockOfWave/bubbles_champagne)
![GitHub pull requests](https://img.shields.io/github/issues-pr/ShockOfWave/bubbles_champagne)
![contributors](https://img.shields.io/github/contributors/ShockOfWave/bubbles_champagne) 
![codesize](https://img.shields.io/github/languages/code-size/ShockOfWave/bubbles_champagne)
![GitHub repo size](https://img.shields.io/github/repo-size/ShockOfWave/bubbles_champagne)
![GitHub top language](https://img.shields.io/github/languages/top/ShockOfWave/bubbles_champagne)
![GitHub language count](https://img.shields.io/github/languages/count/ShockOfWave/bubbles_champagne)

## Overview

This project implements a video classification system using TabNet model and CLIP embeddings. Videos are converted to frames, from which embeddings are extracted using CLIP, followed by classification using TabNet. The system also includes YOLO-based video segmentation for preprocessing.

## Features

- Video segmentation using YOLO model for object detection
- Frame extraction from videos with automatic processing
- CLIP embeddings extraction from images
- TabNet model pretraining and training
- Support for training on any label extracted from directory structure
- LabelEncoder support for converting string labels to numeric values
- Model pretraining capability for improved classification accuracy

## Data

You can download the raw dataset from [here](https://storage.yandexcloud.net/bubbles-champagne/raw_data.7z)

## Project Structure
```
project_root/
├── src/
│   ├── __init__.py                # Package initialization
│   ├── train.py                   # Main training logic
│   ├── data/
│   │   ├── __init__.py           # Data processing initialization
│   │   ├── preprocess.py         # Image preprocessing and embedding extraction
│   │   ├── video_segmentation.py # YOLO-based video segmentation
│   │   ├── crop_frames.py        # Video frame extraction
│   │   └── split_videos.py       # Data splitting into train/val/test
│   ├── models/
│   │   ├── __init__.py           # Models initialization
│   │   ├── main_model.py         # TabNet model and inference methods
│   │   └── clip_inference.py     # CLIP model for embeddings
│   └── utils/
│       ├── __init__.py           # Utilities initialization
│       ├── metrics.py            # Evaluation metrics
│       └── paths.py              # Path handling utilities
├── examples/
│   └── inference.ipynb           # Model inference example
├── main.py                       # Entry point for training
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

## Installation

1. Clone the repository

```bash
git clone https://github.com/ShockOfWave/bubbles_champagne.git
cd bubbles_champagne
```

2. Install dependencies

Before starting, make sure you have all the necessary dependencies installed:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Verify installation

Ensure all dependencies are installed correctly and the project is ready to use. This project requires Python 3.8 or higher.

## Usage

1. Running the training

Run main.py to train the model. Make sure to provide correct paths to data and model saving directory:

```bash
python main.py --root_dir /path/to/data \
               --checkpoints /path/to/checkpoints
```

Parameters:
- `--root_dir`: Path to the data directory
- `--checkpoints`: Path for saving model checkpoints

Task Description:
```
- Task #1: Binary classification of champagne type (pink/white)
- Task #2: Binary classification of container type (plastic/glass)
```

2. Video Processing Pipeline

The system follows these steps:
1. Videos are processed with YOLO segmentation model
2. Processed videos are saved to 'analyzed_videos' directory
3. Videos are split into train/val/test sets
4. Frames are extracted from videos
5. CLIP embeddings are extracted from frames
6. TabNet model is trained on the embeddings

3. Inference

For inference on new data, use [examples/inference.ipynb](https://github.com/ShockOfWave/bubbles_champagne/blob/main/examples/inference.ipynb) to load the saved model and perform classification on new images.

## License

This project is licensed under the MIT License.
