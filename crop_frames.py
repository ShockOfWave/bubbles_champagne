import cv2
import os
from tqdm import tqdm

# Создаем директорию 'frames', если она еще не существует
if not os.path.exists('frames'):
    os.makedirs('frames')

# Функция для нарезки кадров видео
def save_frames(video_path, frames_dir, base_name, fps=10):
    video = cv2.VideoCapture(video_path)  # Открытие видеофайла
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Получение количества кадров в видео

    # Извлечение кадров с заданной частотой
    for i in range(0, frame_count, int(fps)):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)  # Переход к нужному кадру
        ret, frame = video.read()
        if ret:
            # Формируем имя файла кадра, включая исходное имя файла и номер кадра
            frame_filename = os.path.join(frames_dir, f"{base_name}_{frame_count:05}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

    video.release()

# Основной цикл, который проходит через все видео в папке 'data'
for person_name in os.listdir('raw_data'):
    person_dir = os.path.join('raw_data', person_name)
    
    # Проверяем, является ли элемент директорией
    if os.path.isdir(person_dir):
        frames_person_dir = os.path.join('frames', person_name)
        
        # Создаем папку для каждого человека в 'frames', если она еще не существует
        if not os.path.exists(frames_person_dir):
            os.makedirs(frames_person_dir)
        
        for video_file in tqdm(os.listdir(person_dir), desc=f"Processing {person_name}: "):
            video_path = os.path.join(person_dir, video_file)
            # Получаем имя файла без расширения для использования в имени кадра
            base_name, _ = os.path.splitext(video_file)
            
            if video_path.endswith('.mp4') and os.path.isfile(video_path):
                save_frames(video_path, frames_person_dir, base_name)