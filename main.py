import os
import argparse
from glob import glob
from src.utils.split_videos import split_videos
from src.utils.crop_frames import process_videos
from src.train import train_and_evaluate
from src.utils.preprocess import load_label_encoder

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

    train_paths = glob("frames/train/*/*.jpg")
    val_paths = glob("frames/val/*/*.jpg")
    test_paths = glob("frames/test/*/*.jpg")


    print("Training model for task 1...")
    train_and_evaluate(train_paths, val_paths, test_paths, output_dir=args.checkpoints, label_index=0)

    print("Training model for task 2...")
    train_and_evaluate(train_paths, val_paths, test_paths, output_dir=args.checkpoints, label_index=1)

    print("Training model for task 3...")
    train_and_evaluate(train_paths, val_paths, test_paths, output_dir=args.checkpoints, label_index=2)


if __name__ == "__main__":
    main()
