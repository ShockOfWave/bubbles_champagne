import os
from glob import glob
from ultralytics import YOLO
import cv2

# Путь к корню проекта
PROJECT_ROOT = "project_root"
MODEL_PATH = "path/to/your/yolov8-seg-model.pt"

# Инициализация модели
model = YOLO(MODEL_PATH)

def process_images_in_directory(directory_path):
    """
    Функция для обработки всех изображений в указанной директории и ее подпапках.
    """
    image_paths = glob(os.path.join(directory_path, "**", "*.jpg"), recursive=True) + \
                  glob(os.path.join(directory_path, "**", "*.jpeg"), recursive=True) + \
                  glob(os.path.join(directory_path, "**", "*.png"), recursive=True)
    
    for image_path in image_paths:
        if not process_image(image_path):
            os.remove(image_path)
            print(f"Удалено изображение: {image_path}")

def process_image(image_path):
    """
    Функция для передачи изображения в модель и проверки предсказаний.
    Возвращает True, если предсказания не пустые, иначе False.
    """
    # Загрузка изображения
    image = cv2.imread(image_path)
    
    # Получение предсказаний от модели
    results = model(image)
    
    # Проверка на наличие предсказаний
    for result in results:  # Обход всех результатов (для каждого изображения может быть несколько предсказаний)
        if len(result.masks) > 0:  # Если предсказания есть, возвращаем True
            return True
    
    # Если предсказаний нет, возвращаем False
    return False

def process_project_directories():
    """
    Финальная функция для обработки директорий train, val и test.
    """
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(PROJECT_ROOT, split)
        if os.path.exists(split_path):
            process_images_in_directory(split_path)
        else:
            print(f"Директория {split_path} не найдена.")

if __name__ == "__main__":
    process_project_directories()