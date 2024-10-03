import cv2
import torch
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from sklearn.preprocessing import LabelEncoder
import pickle

# Определение устройства для вычислений
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка модели CLIP и процессора для предобработки изображений
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

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

def preprocess_data(paths, label_index, label_encoder=None, save_encoder=False, encoder_path=None):
    """
    Предобработка изображений и создание эмбеддингов.
    Извлекает эмбеддинги для каждого изображения и лейбл по указанному индексу.
    
    Параметры:
    - label_encoder: если передан обученный LabelEncoder, он будет использован для конвертации лейблов.
    - save_encoder: если True, LabelEncoder будет сохранен в файл.
    - encoder_path: путь для сохранения или загрузки LabelEncoder.
    """
    X = []
    y = []

    # Если энкодер не передан, инициализируем новый
    if label_encoder is None:
        label_encoder = LabelEncoder()

    for path in tqdm(paths, desc="Processing images"):
        # Извлечение эмбеддингов
        embeddings = extract_embeddings(path)
        X.append(embeddings)
        
        # Извлечение лейбла по индексу
        label = extract_labels_from_path(path, label_index)
        y.append(label)
    
    # Преобразуем строковые лейблы в числовой формат с помощью LabelEncoder
    y = label_encoder.fit_transform(y)
    
    # Сохраняем энкодер, если указано
    if save_encoder and encoder_path:
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"Label encoder saved to {encoder_path}")
    
    return np.array(X), np.array(y), label_encoder

def load_label_encoder(encoder_path):
    """
    Загрузка LabelEncoder из файла.
    """
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"Label encoder loaded from {encoder_path}")
    return label_encoder