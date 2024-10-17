import torch
import numpy as np
from src.model import VideoClassifier
from src.utils.preprocess import preprocess_data

def inference(model_path, task_number, test_paths):
    """
    Функция для инференса модели.

    Аргументы:
    - model_path: путь к архиву модели, который включает веса модели, предтренер и словарь decode.
    - task_number: номер задачи (1, 2, или 3).
    - test_paths: список путей к изображениям для инференса.

    Возвращает:
    - predicted_labels: список предсказанных текстовых меток.
    """
    model = VideoClassifier()
    model.load_model(model_path, task_number)  

    print(f"Preprocessing test data for task {task_number}...")
    X_test, _ = preprocess_data(test_paths, task_number)  

    print(f"Running inference for task {task_number}...")
    y_pred = model.predict(X_test)  

    predicted_labels = [model.decode[pred] for pred in y_pred]

    return predicted_labels
