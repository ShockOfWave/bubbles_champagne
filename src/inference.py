import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from src.model import VideoClassifier
from src.utils.preprocess import preprocess_data

def inference(model_path, task_number, test_paths, real_labels=None):
    """
    Функция для инференса модели с выводом confusion matrix.

    Аргументы:
    - model_path: путь к архиву модели, который включает веса модели, предтренер и словарь decode.
    - task_number: номер задачи (1, 2, или 3).
    - test_paths: список путей к изображениям для инференса.
    - real_labels: реальные метки (если доступны) для построения confusion matrix.

    Возвращает:
    - predicted_labels: список предсказанных текстовых меток.
    """
    # Шаг 1: Загрузка модели и предтренера для выбранной задачи
    model = VideoClassifier()
    model.load_model(model_path, task_number)  # Загружаем модель, предтренер и decode из архива

    # Шаг 2: Предобработка тестовых данных
    print(f"Preprocessing test data for task {task_number}...")
    X_test, _ = preprocess_data(test_paths, task_number)  # Предобработка данных для выбранной задачи

    # Шаг 3: Предсказание меток
    print(f"Running inference for task {task_number}...")
    y_pred = model.predict(X_test)  # Получаем предсказанные метки (в числовом формате)

    # Шаг 4: Декодирование предсказанных меток в текстовый формат
    predicted_labels = [model.decode[pred] for pred in y_pred]

    # Шаг 5: Вывод confusion matrix (если доступны реальные метки)
    if real_labels:
        real_labels_numeric = [model.decode.index(label) for label in real_labels]  # Преобразование в числовой формат
        y_pred_numeric = y_pred  # Уже в числовом формате

        # Вычисление confusion matrix
        cm = confusion_matrix(real_labels_numeric, y_pred_numeric)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.decode.values())
        
        # Отображение confusion matrix
        plt.figure(figsize=(8, 6))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix for Task {task_number}")
        plt.show()

    return predicted_labels
