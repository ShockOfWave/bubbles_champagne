import os
import shutil
import random
import argparse
from tqdm import tqdm

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def split_videos(root_dir, output_dir, train_split=0.6, val_split=0.3, test_split=0.1):
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)

        if os.path.isdir(class_dir):
            videos = [f for f in os.listdir(class_dir) if f.endswith('.mp4')]
            random.shuffle(videos)
            total_videos = len(videos)
            train_size = int(total_videos * train_split)
            val_size = int(total_videos * val_split)

            train_videos = videos[:train_size]
            val_videos = videos[train_size:train_size + val_size]
            test_videos = videos[train_size + val_size:]

            for split, video_list in zip(['train', 'test', 'val'], [train_videos, val_videos, test_videos]):
                split_class_dir = os.path.join(output_dir, split, class_name)
                create_dir_if_not_exists(split_class_dir)

                for video in tqdm(video_list, desc=f"Copying {split}/{class_name}"):
                    src_video_path = os.path.join(class_dir, video)
                    dst_video_path = os.path.join(split_class_dir, video)
                    shutil.copy(src_video_path, dst_video_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split videos into train, val, and test sets.")
    parser.add_argument('--root_dir', type=str, required=True, help="Path to the root directory containing video classes.")
    parser.add_argument('--output_dir', type=str, default='output', help="Path to the directory where split videos will be saved.")
    parser.add_argument('--train_split', type=float, default=0.6, help="Proportion of videos to use for training.")
    parser.add_argument('--val_split', type=float, default=0.3, help="Proportion of videos to use for validation.")
    parser.add_argument('--test_split', type=float, default=0.1, help="Proportion of videos to use for testing.")
    
    args = parser.parse_args()

    split_videos(args.root_dir, args.output_dir, args.train_split, args.val_split, args.test_split)