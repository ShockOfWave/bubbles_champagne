import os
import shutil
import random
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def create_dir_if_not_exists(directory):
    """
    Creates a directory if it does not exist.

    Parameters
    ----------
    directory: str
        Path to the directory to create.

    Returns
    -------
    None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def split_videos(root_dir, output_dir, train_split=0.7, val_split=0.2, test_split=0.1):
    """
    Splits videos in root_dir into train, val and test sets and saves them in output_dir.

    Parameters
    ----------
    root_dir: str
        Path to the root directory containing video classes.
    output_dir: str
        Path to the output directory where the split videos will be saved.
    train_split: float, optional
        Proportion of videos for training. Default is 0.6.
    val_split: float, optional
        Proportion of videos for validation. Default is 0.2.
    test_split: float, optional
        Proportion of videos for testing. Default is 0.2.

    Returns
    -------
    None
    """
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)

        if os.path.isdir(class_dir):
            videos = [f for f in os.listdir(class_dir) if f.endswith('.mp4')]
            # Shuffle videos randomly
            random.shuffle(videos)

            # First split into train and temp
            train_videos, temp_videos = train_test_split(
                videos, test_size=(1 - train_split), random_state=42
            )
            # Then split temp into val and test
            val_videos, test_videos = train_test_split(
                temp_videos, test_size=(test_split / (test_split + val_split)), random_state=32
            )

            for split, video_list in zip(['train', 'val', 'test'], [train_videos, val_videos, test_videos]):
                split_class_dir = os.path.join(output_dir, split, class_name)
                create_dir_if_not_exists(split_class_dir)

                for video in tqdm(video_list, desc=f"Copying {split}/{class_name}"):
                    src_video_path = os.path.join(class_dir, video)
                    dst_video_path = os.path.join(split_class_dir, video)
                    shutil.copy(src_video_path, dst_video_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split videos into train, val and test sets.")
    parser.add_argument('--root_dir', type=str, required=True, help="Path to the root directory containing video classes.")
    parser.add_argument('--output_dir', type=str, default='output', help="Path to the directory where the split videos will be saved.")
    parser.add_argument('--train_split', type=float, default=0.6, help="Proportion of videos for training.")
    parser.add_argument('--val_split', type=float, default=0.3, help="Proportion of videos for validation.")
    parser.add_argument('--test_split', type=float, default=0.1, help="Proportion of videos for testing.")
    
    args = parser.parse_args()

    split_videos(args.root_dir, args.output_dir, args.train_split, args.val_split, args.test_split)
