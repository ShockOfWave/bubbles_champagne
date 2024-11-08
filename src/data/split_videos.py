import os
import shutil
import random
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def create_dir_if_not_exists(directory):
    """
    Создает директорию, если она не существует.

    Параметры
    ----------
    directory: str
        Путь к директории для создания.

    Возвращает
    -------
    None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def split_videos(root_dir, output_dir, train_split=0.7, val_split=0.2, test_split=0.1):
    """
    Разбивает видео в root_dir на train, val и test наборы и сохраняет их в output_dir.

    Параметры
    ----------
    root_dir: str
        Путь к корневой директории, содержащей классы видео.
    output_dir: str
        Путь к выходной директории, где будут сохранены разделенные видео.
    train_split: float, optional
        Доля видео для обучения. По умолчанию 0.6.
    val_split: float, optional
        Доля видео для валидации. По умолчанию 0.2.
    test_split: float, optional
        Доля видео для тестирования. По умолчанию 0.2.

    Возвращает
    -------
    None
    """
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)

        if os.path.isdir(class_dir):
            videos = [f for f in os.listdir(class_dir) if f.endswith('.mp4')]
            # Размешиваем видео случайным образом
            random.shuffle(videos)

            # Сначала делим на train и temp
            train_videos, temp_videos = train_test_split(
                videos, test_size=(1 - train_split), random_state=42
            )
            # Затем делим temp на val и test
            val_videos, test_videos = train_test_split(
                temp_videos, test_size=(test_split / (test_split + val_split)), random_state=42
            )

            for split, video_list in zip(['train', 'val', 'test'], [train_videos, val_videos, test_videos]):
                split_class_dir = os.path.join(output_dir, split, class_name)
                create_dir_if_not_exists(split_class_dir)

                for video in tqdm(video_list, desc=f"Copying {split}/{class_name}"):
                    src_video_path = os.path.join(class_dir, video)
                    dst_video_path = os.path.join(split_class_dir, video)
                    shutil.copy(src_video_path, dst_video_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Разделите видео на train, val и test наборы.")
    parser.add_argument('--root_dir', type=str, required=True, help="Путь к корневой директории, содержащей классы видео.")
    parser.add_argument('--output_dir', type=str, default='output', help="Путь к директории, где будут сохранены разделенные видео.")
    parser.add_argument('--train_split', type=float, default=0.6, help="Доля видео для обучения.")
    parser.add_argument('--val_split', type=float, default=0.3, help="Доля видео для валидации.")
    parser.add_argument('--test_split', type=float, default=0.1, help="Доля видео для тестирования.")
    
    args = parser.parse_args()

    split_videos(args.root_dir, args.output_dir, args.train_split, args.val_split, args.test_split)
