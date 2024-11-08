from src.models.main_model import VideoClassifier
from src.data.preprocess import preprocess_data
from src.utils.config import decode
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve, accuracy_score
import seaborn as sns
import numpy as np
from sklearn.preprocessing import label_binarize

def plot_and_save_roc_auc(y_true, y_pred, output_dir, task_number):
    # Преобразование строковых меток в числовые (бинарные)
    unique_labels = list(set(y_true))
    y_true_bin = [unique_labels.index(label) for label in y_true]
    y_pred_bin = [unique_labels.index(label) for label in y_pred]

    fpr, tpr, _ = roc_curve(y_true_bin, y_pred_bin, pos_label=1)
    roc_auc = roc_auc_score(y_true_bin, y_pred_bin)
    
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, f"roc_auc_curve_task{task_number}.png"))
    plt.close()


def plot_and_save_confusion_matrix(y_true, y_pred, output_dir, task_number, decode_labels):
    cm = confusion_matrix(y_true, y_pred, normalize="true")  # Нормализуем confusion matrix
    plt.figure()
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=decode_labels, yticklabels=decode_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Normalized Confusion Matrix")
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_task{task_number}.png"))
    plt.close()


def plot_and_save_precision_recall(y_true, y_pred, output_dir, task_number):
    # Преобразуем строковые метки в числовые (бинарные)
    unique_labels = list(set(y_true))
    y_true_bin = [unique_labels.index(label) for label in y_true]
    y_pred_bin = [unique_labels.index(label) for label in y_pred]

    precision, recall, _ = precision_recall_curve(y_true_bin, y_pred_bin, pos_label=1)
    
    plt.figure()
    plt.plot(recall, precision, color="b", label="Precision-Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(output_dir, f"precision_recall_curve_task{task_number}.png"))
    plt.close()


def train_and_evaluate(train_paths, val_paths, test_paths, output_dir, task_number, n_d=64, n_a=64, n_steps=5, gamma=1.5, lambda_sparse=1e-4, lr=2e-2, step_size=10, gamma_lr=0.9, batch_size=128, virtual_batch_size=256, patience=30, pretrain_ratio=0.8):
    """
    Функция для обучения модели и оценки на тестовых данных.
    """
    if isinstance(train_paths, list):
        print("Preprocessing training data...")
        X_train, y_train = preprocess_data(train_paths, task_number, data_folder='data/train/')
    elif train_paths.endswith('.pkl'):
        with open(train_paths, 'rb') as f:
            X_train, y_train = pickle.load(f)
    else:
        raise ValueError("Invalid input type. Expected 'str' or 'list'")
    
    if isinstance(val_paths, list):
        print("Preprocessing validation data...")
        X_val, y_val = preprocess_data(val_paths, task_number, data_folder='data/val/')
    elif val_paths.endswith('.pkl'):
        with open(val_paths, 'rb') as f:
            X_val, y_val = pickle.load(f)
    else:
        raise ValueError("Invalid input type. Expected 'str' or 'list'")
    
    if isinstance(test_paths, list):
        print("Preprocessing testing data...")
        X_test, y_test = preprocess_data(test_paths, task_number, data_folder='data/test/')
    elif test_paths.endswith('.pkl'):
        with open(test_paths, 'rb') as f:
            X_test, y_test = pickle.load(f)
    else:
        raise ValueError("Invalid input type. Expected 'str' or 'list'")
    

    # Создаем директорию для сохранения модели, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Создаем экземпляр модели
    model = VideoClassifier(
        n_d=n_d, 
        n_a=n_a, 
        n_steps=n_steps, 
        gamma=gamma, 
        lambda_sparse=lambda_sparse, 
        lr=lr, 
        step_size=step_size, 
        gamma_lr=gamma_lr, 
        batch_size=batch_size, 
        virtual_batch_size=virtual_batch_size,
        max_epochs=1000
    )

    # Устанавливаем словарь decode для задачи
    decode_labels = decode[task_number - 1]
    model.decode = decode_labels

    # Этап предобучения
    print("Starting pretraining...")
    model.pretrain(X_train, X_val, pretrain_ratio=pretrain_ratio)

    # Основное обучение
    print("Starting training...")
    model.train(X_train, y_train, X_val, y_val, patience=patience)

    # Оценка на тестовых данных
    print("Evaluating on test data...")
    y_pred_indices = model.predict(X_test)

    # Преобразование предсказанных меток с использованием decode
    y_pred = [decode_labels[pred] for pred in y_pred_indices]
    y_true = [decode_labels[true] for true in y_test]  # преобразуем и y_test для соответствия с метками

    # Оценка точности с преобразованными метками
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy score on test data: {accuracy:.4f}")
    # Сохранение метрик и графиков
    plot_and_save_roc_auc(y_true, y_pred, output_dir, task_number)
    plot_and_save_confusion_matrix(y_true, y_pred, output_dir, task_number, list(decode_labels.values()))
    plot_and_save_precision_recall(y_true, y_pred, output_dir, task_number)

    # Сохранение модели, предтренера и словаря decode
    model_path = os.path.join(output_dir, f"trained_model_task{task_number}")
    model.save_model(model_path)
    
    return accuracy
