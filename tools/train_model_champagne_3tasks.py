import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from glob import glob
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
from embeds_generator import extract_embeddings
import argparse

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

def extract_embeddings_from_paths(paths):
    embeddings = []
    for path in tqdm(paths, desc="Generating embeddings"):
        embeddings.append(extract_embeddings(path))
    return np.array(embeddings)

def save_embeddings(X_train, X_val, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pickle.dump(X_train, open(f"{output_dir}/champagne_train.pkl", "wb"))
    pickle.dump(X_val, open(f"{output_dir}/champagne_val.pkl", "wb"))
    print("Embeddings saved!")

def generate_labels(paths, label_index):
    y = []
    for path in tqdm(paths, desc="Generating labels"):
        path_parts = path.split('/')
        labels = [converter[x] for x in path_parts[-2].split('_')]
        y.append(labels[label_index])
    return np.array(y)

def pretrain_model(X_train, X_val, pretrain_ratio, pretrain_lr, pretrain_mask_type, pretrain_verbose):
    unsupervised_model = TabNetPretrainer(
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=pretrain_lr),
        mask_type=pretrain_mask_type,
        verbose=pretrain_verbose
    )

    unsupervised_model.fit(
        X_train=X_train,
        eval_set=[X_val],
        pretraining_ratio=pretrain_ratio,
    )
    return unsupervised_model

def train_task(X_train, y_train, X_val, y_val, unsupervised_model, task_name, output_dir, 
               n_d, n_a, n_steps, gamma, lambda_sparse, optimizer_fn, optimizer_params, 
               scheduler_params, scheduler_fn, patience, batch_size, virtual_batch_size, metric_list, verbose):

    clf = TabNetClassifier(
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        lambda_sparse=lambda_sparse,
        optimizer_fn=optimizer_fn,
        optimizer_params=optimizer_params,
        scheduler_params=scheduler_params,
        scheduler_fn=scheduler_fn,
        verbose=verbose
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=metric_list,
        patience=patience,
        from_unsupervised=unsupervised_model,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    saving_path_name = f"{output_dir}/{task_name}_model"
    clf.save_model(saving_path_name)
    print(f"Model for {task_name} saved!")

def main(args):
    train_paths = glob(os.path.join(args.train_dir, '*/*.jpg'))
    val_paths = glob(os.path.join(args.val_dir, '*/*.jpg'))

    X_train = extract_embeddings_from_paths(train_paths)
    X_val = extract_embeddings_from_paths(val_paths)

    save_embeddings(X_train, X_val, args.output_dir)

    unsupervised_model = pretrain_model(X_train, X_val, args.pretrain_ratio, args.pretrain_lr, args.pretrain_mask_type, args.pretrain_verbose)

    if 'task1' in args.task:
        y_train_task1 = generate_labels(train_paths, 0)
        y_val_task1 = generate_labels(val_paths, 0)
        train_task(X_train, y_train_task1, X_val, y_val_task1, unsupervised_model, "task1", args.output_dir,
                   args.n_d, args.n_a, args.n_steps, args.gamma, args.lambda_sparse, torch.optim.Adam, 
                   dict(lr=args.lr), {"step_size": args.step_size, "gamma": args.gamma_lr}, torch.optim.lr_scheduler.StepLR, 
                   args.patience, args.batch_size, args.virtual_batch_size, ["accuracy", "balanced_accuracy"], args.verbose)

    if 'task2' in args.task:
        y_train_task2 = generate_labels(train_paths, 1)
        y_val_task2 = generate_labels(val_paths, 1)
        train_task(X_train, y_train_task2, X_val, y_val_task2, unsupervised_model, "task2", args.output_dir,
                   args.n_d, args.n_a, args.n_steps, args.gamma, args.lambda_sparse, torch.optim.Adam, 
                   dict(lr=args.lr), {"step_size": args.step_size, "gamma": args.gamma_lr}, torch.optim.lr_scheduler.StepLR, 
                   args.patience, args.batch_size, args.virtual_batch_size, ["auc", "accuracy", "balanced_accuracy"], args.verbose)

    if 'task3' in args.task:
        y_train_task3 = generate_labels(train_paths, 2)
        y_val_task3 = generate_labels(val_paths, 2)
        train_task(X_train, y_train_task3, X_val, y_val_task3, unsupervised_model, "task3", args.output_dir,
                   args.n_d, args.n_a, args.n_steps, args.gamma, args.lambda_sparse, torch.optim.Adam, 
                   dict(lr=args.lr), {"step_size": args.step_size, "gamma": args.gamma_lr}, torch.optim.lr_scheduler.StepLR, 
                   args.patience, args.batch_size, args.virtual_batch_size, ["accuracy", "balanced_accuracy"], args.verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TabNet model on embeddings for multiple tasks.")

    # Аргументы для путей к данным
    parser.add_argument('--train_dir', type=str, required=True, help="Path to the training images.")
    parser.add_argument('--val_dir', type=str, required=True, help="Path to the validation images.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to save the models and embeddings.")
    
    # Аргумент для выбора задачи
    parser.add_argument('--task', type=str, nargs='+', choices=['task1', 'task2', 'task3'], required=True, help="Tasks to run: choose from 'task1', 'task2', 'task3'.")

    # Аргументы для предобучения
    parser.add_argument('--pretrain_ratio', type=float, default=0.8, help="Ratio for pretraining (default: 0.8).")
    parser.add_argument('--pretrain_lr', type=float, default=2e-2, help="Learning rate for pretraining (default: 2e-2).")
    parser.add_argument('--pretrain_mask_type', type=str, default="entmax", help="Mask type for pretraining (default: entmax).")
    parser.add_argument('--pretrain_verbose', type=int, default=10, help="Verbosity level for pretraining (default: 10).")

    # Гиперпараметры для обучения TabNet
    parser.add_argument('--n_d', type=int, default=64, help="Number of features for decision layers (default: 64).")
    parser.add_argument('--n_a', type=int, default=64, help="Number of features for attention layers (default: 64).")
    parser.add_argument('--n_steps', type=int, default=5, help="Number of decision steps (default: 5).")
    parser.add_argument('--gamma', type=float, default=1.5, help="Gamma value for TabNet (default: 1.5).")
    parser.add_argument('--lambda_sparse', type=float, default=1e-4, help="Sparse regularization lambda (default: 1e-4).")
    
    # Оптимизатор и обучение
    parser.add_argument('--lr', type=float, default=2e-2, help="Learning rate for training (default: 2e-2).")
    parser.add_argument('--step_size', type=int, default=10, help="Step size for learning rate scheduler (default: 10).")
    parser.add_argument('--gamma_lr', type=float, default=0.9, help="Gamma for learning rate scheduler (default: 0.9).")
    parser.add_argument('--patience', type=int, default=30, help="Patience for early stopping (default: 30).")
    
    # Batch size
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training (default: 128).")
    parser.add_argument('--virtual_batch_size', type=int, default=256, help="Virtual batch size for TabNet (default: 256).")
    
    # Прочие параметры
    parser.add_argument('--verbose', type=int, default=10, help="Verbosity level for training (default: 10).")

    args = parser.parse_args()
    main(args)