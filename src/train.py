from src.model import VideoClassifier
from src.utils.preprocess import preprocess_data
import os

def train_and_evaluate(train_paths, val_paths, test_paths, output_dir, task_number, n_d=64, n_a=64, n_steps=5, gamma=1.5, lambda_sparse=1e-4, lr=2e-2, step_size=10, gamma_lr=0.9, batch_size=128, virtual_batch_size=256, patience=30, pretrain_ratio=0.8):
    """
    Функция для обучения модели и оценки на тестовых данных.
    
    Параметры:
    - train_paths, val_paths, test_paths: списки путей к тренировочным, валидационным и тестовым данным.
    - label_index: индекс лейбла, который используется для обучения.
    - n_d, n_a, n_steps, gamma, lambda_sparse, lr, step_size, gamma_lr, batch_size, virtual_batch_size, patience: гиперпараметры модели.
    - pretrain_ratio: соотношение данных для предобучения.
    """
    # Предобработка данных и сохранение/обучение энкодера
    print("Preprocessing training data...")
    X_train, y_train = preprocess_data(train_paths, task_number)

    print("Preprocessing validation data...")
    X_val, y_val = preprocess_data(val_paths, task_number)

    print("Preprocessing test data...")
    X_test, y_test = preprocess_data(test_paths, task_number)

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
        virtual_batch_size=virtual_batch_size
    )

    # Этап предобучения
    print("Starting pretraining...")
    model.pretrain(X_train, X_val, pretrain_ratio=pretrain_ratio)

    # Основное обучение
    print("Starting training...")
    model.train(X_train, y_train, X_val, y_val, patience=patience)

    # Оценка на тестовых данных
    print("Evaluating on test data...")
    accuracy = model.evaluate(X_test, y_test)

    # Сохранение модели
    model_path = os.path.join(output_dir, f"trained_model_task{task_number}.pkl")
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    return accuracy