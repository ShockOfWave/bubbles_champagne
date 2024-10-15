import cv2
import torch
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import pickle

# Определение устройства для вычислений
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка модели CLIP и процессора для предобработки изображений
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

converter = {
            "pink": 0,
            "white": 1,
            "plastic": 0,
            "glass": 1,
            "0": 0,
            "10": 1,
            "15": 2,
            "20": 3,
            "min": 0
            }

decode = {
    0: {
        0: "pink",
        1: "white",
    },
    1: {
        0: "plastic",
        1: "glass",
    },
    2: {
        0: "0",
        1: "10",
        2: "15",
        3: "20",
    }
}

def extract_embeddings(img_path):
    """
    Извлечение эмбеддингов из изображения с помощью модели CLIP.
    """
    image = cv2.imread(img_path)
    inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embed = model.get_image_features(**inputs).cpu().squeeze(0).numpy()
    return embed

def extract_labels_from_path(path, label_index):
    """
    Извлекает все лейблы из пути к изображению и возвращает лейбл по указанному индексу.
    Путь должен быть в формате: /data/{train/val/test}/{label1}_{label2}_{label3}_{label n}/img.jpeg
    """
    # Разделяем путь на части
    label_part = path.split('/')[-2]  # Извлекаем часть с лейблами (перед изображением)
    labels = label_part.split('_')  # Разделяем по символу "_"
    
    # Проверяем, что индекс валиден
    if label_index >= len(labels):
        raise ValueError(f"Неверный индекс лейбла: {label_index}. В пути {path} всего {len(labels)} лейблов.")
    
    # Возвращаем лейбл по указанному индексу
    return labels[label_index]

def preprocess_data(paths, task_number):
    """
    Предобработка изображений и создание эмбеддингов.
    Извлекает эмбеддинги для каждого изображения и лейбл по указанному индексу.
    """

    label_index = task_number - 1

    X = []
    y = []


    for path in tqdm(paths, desc="Processing images"):
        # Извлечение эмбеддингов
        embeddings = extract_embeddings(path)
        X.append(embeddings)
        
        # Извлечение лейбла по индексу
        label = extract_labels_from_path(path, label_index)
        y.append(label)
    
    # Преобразуем строковые лейблы в числовой формат
    y = [converter[x] for x in y]
    
    return np.array(X), np.array(y)
