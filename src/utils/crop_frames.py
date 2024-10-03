import cv2
import os
from tqdm import tqdm
import argparse

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_frames(video_path, frames_dir, base_name, fps=10):
    video = cv2.VideoCapture(video_path)  
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  

    for i in range(0, frame_count, int(fps)):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)  
        ret, frame = video.read()
        if ret:
            frame_filename = os.path.join(frames_dir, f"{base_name}_{i:05}.jpg")
            cv2.imwrite(frame_filename, frame)

    video.release()

def process_videos(root_dir, output_dir, fps=10):
    for split in ['train', 'test', 'val']:
        split_dir = os.path.join(root_dir, split)
        
        if not os.path.exists(split_dir):
            print(f"Directory {split_dir} does not exist. Skipping...")
            continue

        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)

            if os.path.isdir(class_dir):
                frames_class_dir = os.path.join(output_dir, split, class_name)
                create_dir_if_not_exists(frames_class_dir)

                for video_file in tqdm(os.listdir(class_dir), desc=f"Processing {split}/{class_name}: "):
                    video_path = os.path.join(class_dir, video_file)
                    base_name, _ = os.path.splitext(video_file)

                    if video_path.endswith('.mp4') and os.path.isfile(video_path):
                        save_frames(video_path, frames_class_dir, base_name, fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument('--root_dir', type=str, required=True, help="Path to the root directory containing video classes.")
    parser.add_argument('--output_dir', type=str, default='frames', help="Path to the directory where frames will be saved.")
    parser.add_argument('--fps', type=int, default=10, help="Frames per second to extract from video.")
    
    args = parser.parse_args()

    create_dir_if_not_exists(args.output_dir)
    
    process_videos(args.root_dir, args.output_dir, args.fps)