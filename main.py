import os
import argparse
from glob import glob
from src.data.split_videos import split_videos
from src.data.crop_frames import process_videos
from src.train import train_and_evaluate
from src.data.video_segmentation import VideoSegmenter
from tqdm import tqdm

import torch
import numpy as np
import random

def set_seed(seed: int):
    random.seed(seed)                      # For random module
    np.random.seed(seed)                   # For numpy
    torch.manual_seed(seed)                # For PyTorch CPU
    torch.cuda.manual_seed(seed)           # For PyTorch GPU
    torch.cuda.manual_seed_all(seed)       # For all GPUs (if multiple are used)
    torch.backends.cudnn.deterministic = True  # For deterministic behavior
    torch.backends.cudnn.benchmark = False     # Disable optimizations that make the process random

def process_videos_with_segmentation(input_dir, output_dir):
    """Process videos with segmentation model and save results"""
    os.makedirs(output_dir, exist_ok=True)
    segmenter = VideoSegmenter()
    
    # Get all video files
    video_files = glob(os.path.join(input_dir, "**", "*.mp4"), recursive=True)
    
    # Process each video with progress bar for directories
    for video_file in tqdm(video_files, desc="Processing directories", unit="video"):
        relative_path = os.path.relpath(video_file, input_dir)
        # Add _analyzed suffix before the extension
        base, ext = os.path.splitext(relative_path)
        output_path = os.path.join(output_dir, f"{base}_analyzed{ext}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        segmenter.process_video(video_file, output_path)

def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description="Video classification training")
    
    parser.add_argument('--root_dir', type=str, required=True, help="Path to the train data directory")
    parser.add_argument('--checkpoints', type=str, required=True, help="Directory to save the model and results")

    args = parser.parse_args()

    # Process videos with segmentation and save to analyzed_videos directory
    analyzed_dir = "analyzed_videos"
    if not os.path.exists(analyzed_dir):
        process_videos_with_segmentation(args.root_dir, analyzed_dir)
    else:
        print("Videos already analyzed")

    if not os.path.exists("data_split/"):
        split_videos(root_dir=analyzed_dir, output_dir="data_split")
    else:
        print("Data split already exists. Skipping data split...")

    if not os.path.exists("frames/"):
        process_videos(root_dir="data_split", output_dir="frames", fps=10)
    else:
        print("Frames already extracted. Skipping frame extraction...")


    if os.path.exists('data/train/data_task1.pkl'):
        print("Using preprocessed train data")
        train_paths = "data/train/data_task1.pkl"
    else:
        train_paths = glob("frames/train/*/*.jpg")
    
    if os.path.exists('data/val/data_task1.pkl'):
        print("Using preprocessed val data")
        val_paths = "data/val/data_task1.pkl"
    else:
        val_paths = glob("frames/val/*/*.jpg")


    if os.path.exists('data/test/data_task1.pkl'):
        print("Using preprocessed test data")
        test_paths = "data/test/data_task1.pkl"
    else:
        test_paths = glob("frames/test/*/*.jpg")


    print("Training model for task (pink/white)...")
    train_and_evaluate(train_paths, val_paths, test_paths, output_dir=args.checkpoints, task_number=1,
                       n_d=64, n_a=10, n_steps=5, gamma=0.9, lambda_sparse=1e-3, lr=2e-2, 
                       step_size=10, gamma_lr=0.9, batch_size=1024, virtual_batch_size=256, patience=30, pretrain_ratio=0.3)
    

    if os.path.exists('data/train/data_task2.pkl'):
        print("Using preprocessed train data")
        train_paths = "data/train/data_task2.pkl"
    else:
        train_paths = glob("frames/train/*/*.jpg")
    
    if os.path.exists('data/val/data_task2.pkl'):
        print("Using preprocessed val data")
        val_paths = "data/val/data_task2.pkl"
    else:
        val_paths = glob("frames/val/*/*.jpg")


    if os.path.exists('data/test/data_task2.pkl'):
        print("Using preprocessed test data")
        test_paths = "data/test/data_task2.pkl"
    else:
        test_paths = glob("frames/test/*/*.jpg")

    print("Training model for task (glass/plastic)...")
    train_and_evaluate(train_paths, val_paths, test_paths, output_dir=args.checkpoints, task_number=2,
                       n_d=64, n_a=10, n_steps=5, gamma=0.9, lambda_sparse=1e-3, lr=2e-2, 
                       step_size=10, gamma_lr=0.9, batch_size=1024, virtual_batch_size=256, patience=30, pretrain_ratio=0.3)


if __name__ == "__main__":
    main()