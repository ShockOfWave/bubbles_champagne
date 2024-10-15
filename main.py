import os
import argparse
from glob import glob
from src.utils.split_videos import split_videos
from src.utils.crop_frames import process_videos
from src.train import train_and_evaluate
from src.utils.clear_frames import process_project_directories


def main():
    parser = argparse.ArgumentParser(description="Video classification training")
    
    parser.add_argument('--root_dir', type=str, required=True, help="Path to the train data directory")
    parser.add_argument('--checkpoints', type=str, required=True, help="Directory to save the model and results")
    args = parser.parse_args()

    if not os.path.exists("data_split/"):
        split_videos(root_dir=args.root_dir, output_dir="data_split")
    else:
        print("Data split already exists. Skipping data split...")

    if not os.path.exists("frames/"):
        process_videos(root_dir="data_split", output_dir="frames")
    else:
        print("Frames already extracted. Skipping frame extraction...")
    
    print('Clearing frames...')
    process_project_directories(project_root='frames')
    print('Frames cleared.')

    train_paths = glob("frames/train/*/*.jpg")
    val_paths = glob("frames/val/*/*.jpg")
    test_paths = glob("frames/test/*/*.jpg")


    print("Training model for task (pink/white)...")
    train_and_evaluate(train_paths, val_paths, test_paths, output_dir=args.checkpoints, task_number=1,
                       n_d=64, n_a=64, n_steps=5, gamma=1.5, lambda_sparse=1e-4, lr=2e-2, 
                       step_size=10, gamma_lr=0.9, batch_size=128, virtual_batch_size=256, patience=30, pretrain_ratio=0.8)

    print("Training model for task (glass/plastic)...")
    train_and_evaluate(train_paths, val_paths, test_paths, output_dir=args.checkpoints, task_number=2,
                       n_d=64, n_a=64, n_steps=5, gamma=1.5, lambda_sparse=1e-4, lr=2e-2, 
                       step_size=10, gamma_lr=0.9, batch_size=128, virtual_batch_size=256, patience=30, pretrain_ratio=0.8)

    # Обучение для задачи 3 (time)
    print("Training model for task (time)...")
    train_and_evaluate(train_paths, val_paths, test_paths, output_dir=args.checkpoints, task_number=3,
                       n_d=64, n_a=64, n_steps=5, gamma=1.5, lambda_sparse=1e-4, lr=2e-2, 
                       step_size=20, gamma_lr=0.95, batch_size=128, virtual_batch_size=256, patience=100, pretrain_ratio=0.8)

if __name__ == "__main__":
    main()