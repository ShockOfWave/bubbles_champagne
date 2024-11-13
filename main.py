import os
import argparse
from glob import glob
from src.data.split_videos import split_videos
from src.data.crop_frames import process_videos
from src.train import train_and_evaluate
from src.data.clear_frames import process_project_directories

import torch
import numpy as np
import random

def set_seed(seed: int):
    random.seed(seed)                      # Для модуля random
    np.random.seed(seed)                   # Для numpy
    torch.manual_seed(seed)                # Для CPU в PyTorch
    torch.cuda.manual_seed(seed)           # Для GPU в PyTorch
    torch.cuda.manual_seed_all(seed)       # Для всех GPU (если используется несколько)
    torch.backends.cudnn.deterministic = True  # Для детерминированности
    torch.backends.cudnn.benchmark = False     # Отключить оптимизации, которые делают процесс случайным

def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description="Video classification training")
    
    parser.add_argument('--root_dir', type=str, required=True, help="Path to the train data directory")
    parser.add_argument('--checkpoints', type=str, required=True, help="Directory to save the model and results")
    parser.add_argument('--skip_clearing', type=bool, required=False, help="Skip empty frames clearing stage.", default=False)
    args = parser.parse_args()

    if not os.path.exists("data_split/"):
        split_videos(root_dir=args.root_dir, output_dir="data_split")
    else:
        print("Data split already exists. Skipping data split...")

    if not os.path.exists("frames/"):
        process_videos(root_dir="data_split", output_dir="frames", fps=10)
    else:
        print("Frames already extracted. Skipping frame extraction...")
    
    if not args.skip_clearing:
        print('Clearing frames...')
        process_project_directories(project_root='frames')
        print('Frames cleared.')
    else:
        print("Skip clearing flag is True. Skipping empty frames clearing.")


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


    # if os.path.exists('data/train/data_task3.pkl'):
    #     print("Using preprocessed train data")
    #     train_paths = "data/train/data_task3.pkl"
    # else:
    #     train_paths = glob("frames/train/*/*.jpg")
    
    # if os.path.exists('data/val/data_task3.pkl'):
    #     print("Using preprocessed val data")
    #     val_paths = "data/val/data_task3.pkl"
    # else:
    #     val_paths = glob("frames/val/*/*.jpg")


    # if os.path.exists('data/test/data_task3.pkl'):
    #     print("Using preprocessed test data")
    #     test_paths = "data/test/data_task3.pkl"
    # else:
    #     test_paths = glob("frames/test/*/*.jpg")
    # # Обучение для задачи 3 (time)
    # print("Training model for task (time)...")
    # train_and_evaluate(train_paths, val_paths, test_paths, output_dir=args.checkpoints, task_number=3,
    #                    n_d=64, n_a=10, n_steps=5, gamma=0.9, lambda_sparse=1e-3, lr=2e-2, 
    #                    step_size=10, gamma_lr=0.9, batch_size=1024, virtual_batch_size=256, patience=30, pretrain_ratio=0.3)

if __name__ == "__main__":
    main()