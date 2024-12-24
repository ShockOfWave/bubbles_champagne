from src.models.main_model import VideoClassifier
from src.data.preprocess import preprocess_data
from src.utils.config import decode
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from src.data.plots import plot_and_save_all_metrics


def train_and_evaluate(train_paths, val_paths, test_paths, output_dir, task_number, n_d=64, n_a=64, n_steps=5, gamma=1.5, lambda_sparse=1e-4, lr=2e-2, step_size=10, gamma_lr=0.9, batch_size=128, virtual_batch_size=256, patience=30, pretrain_ratio=0.8):
    """
    Function for training the model and evaluating on test data.
    """
    if isinstance(train_paths, list):
        print("Preprocessing training data...")
        X_train, y_train = preprocess_data(train_paths, task_number, data_folder='data/train/', save_data=True, process_labels=True)
    elif train_paths.endswith('.pkl'):
        with open(train_paths, 'rb') as f:
            X_train, y_train = pickle.load(f)
    else:
        raise ValueError("Invalid input type. Expected 'str' or 'list'")
    
    if isinstance(val_paths, list):
        print("Preprocessing validation data...")
        X_val, y_val = preprocess_data(val_paths, task_number, data_folder='data/val/', save_data=True, process_labels=True)
    elif val_paths.endswith('.pkl'):
        with open(val_paths, 'rb') as f:
            X_val, y_val = pickle.load(f)
    else:
        raise ValueError("Invalid input type. Expected 'str' or 'list'")
    
    if isinstance(test_paths, list):
        print("Preprocessing testing data...")
        X_test, y_test = preprocess_data(test_paths, task_number, data_folder='data/test/', save_data=True, process_labels=True)
    elif test_paths.endswith('.pkl'):
        with open(test_paths, 'rb') as f:
            X_test, y_test = pickle.load(f)
    else:
        raise ValueError("Invalid input type. Expected 'str' or 'list'")
    

    # Create directory for saving model if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create model instance
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

    # Set decode dictionary for the task
    decode_labels = decode[task_number - 1]
    model.decode = decode_labels

    # Pretraining stage
    print("Starting pretraining...")
    model.pretrain(X_train, X_val, pretrain_ratio=pretrain_ratio)

    # Main training
    print("Starting training...")
    model.train(X_train, y_train, X_val, y_val, patience=patience)

    # Evaluate on test data
    print("Evaluating on test data...")
    y_pred_indices = model.predict(X_test)

    # Transform predicted labels using decode
    y_pred = [decode_labels[pred] for pred in y_pred_indices]
    y_true = [decode_labels[true] for true in y_test]  # transform y_test to match labels

    # Evaluate accuracy with transformed labels
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy score on test data: {accuracy:.4f}")
    # Save metrics and plots
    plot_and_save_all_metrics(y_true, y_pred, output_dir, task_number, list(decode_labels.values()))

    # Save model, pretrainer and decode dictionary
    model_path = os.path.join(output_dir, f"trained_model_task{task_number}")
    model.save_model(model_path)
    train_loss = model.model.history["train_cross_entropy"]
    val_loss = model.model.history["val_cross_entropy"]
    
    return accuracy
