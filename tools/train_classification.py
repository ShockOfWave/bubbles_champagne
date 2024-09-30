import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from glob import glob
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
from tools.embeds_generator import extract_embeddings
import argparse

def extract_embeddings_from_paths(paths):
    embeddings = []
    for path in tqdm(paths, desc="Generating embeddings"):
        embeddings.append(extract_embeddings(path))
    return np.array(embeddings)

def prepare_data(datasets_paths):
    """
    Собираем данные из разных наборов данных и присваиваем каждому набору уникальный лейбл.
    datasets_paths - список кортежей вида (path_to_dataset, label).
    """
    X = []
    y = []
    
    for dataset_path, label in datasets_paths:
        image_paths = glob(os.path.join(dataset_path, '*/*.jpg'))
        embeddings = extract_embeddings_from_paths(image_paths)
        X.append(embeddings)
        y += [label] * len(image_paths)

    X = np.concatenate(X, axis=0)
    y = np.array(y)
    return X, y

def pretrain_model(X_train, X_val, args):
    """
    Предобучение модели TabNet для улучшения представлений.
    """
    pretrainer = TabNetPretrainer(
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=args.pretrain_lr),
        mask_type=args.pretrain_mask_type,
        verbose=args.pretrain_verbose
    )

    pretrainer.fit(
        X_train=X_train,
        eval_set=[X_val],
        pretraining_ratio=args.pretrain_ratio,
    )
    return pretrainer

def train_model(X_train, y_train, X_val, y_val, unsupervised_model, args):
    clf = TabNetClassifier(
        n_d=args.n_d,
        n_a=args.n_a,
        n_steps=args.n_steps,
        gamma=args.gamma,
        lambda_sparse=args.lambda_sparse,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=args.lr),
        scheduler_params={"step_size": args.step_size, "gamma": args.gamma_lr},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        verbose=args.verbose
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["accuracy", "balanced_accuracy"],
        patience=args.patience,
        from_unsupervised=unsupervised_model,
        batch_size=args.batch_size,
        virtual_batch_size=args.virtual_batch_size,
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    clf.save_model(f"{args.output_dir}/classification_model")
    print("Model saved!")

def main(args):
    # Подготавливаем данные
    train_datasets = [
        (args.train_dataset1, 0),
        (args.train_dataset2, 1),
        (args.train_dataset3, 2)
    ]
    val_datasets = [
        (args.val_dataset1, 0),
        (args.val_dataset2, 1),
        (args.val_dataset3, 2)
    ]

    X_train, y_train = prepare_data(train_datasets)
    X_val, y_val = prepare_data(val_datasets)

    unsupervised_model = pretrain_model(X_train, X_val, args)

    train_model(X_train, y_train, X_val, y_val, unsupervised_model, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TabNet model for classification across multiple datasets with pretraining.")
    
    parser.add_argument('--train_dataset1', type=str, required=True, help="Path to the first train dataset.")
    parser.add_argument('--train_dataset2', type=str, required=True, help="Path to the second train dataset.")
    parser.add_argument('--train_dataset3', type=str, required=True, help="Path to the third train dataset.")
    
    parser.add_argument('--val_dataset1', type=str, required=True, help="Path to the first validation dataset.")
    parser.add_argument('--val_dataset2', type=str, required=True, help="Path to the second validation dataset.")
    parser.add_argument('--val_dataset3', type=str, required=True, help="Path to the third validation dataset.")
    
    parser.add_argument('--output_dir', type=str, required=True, help="Path to save the trained model.")
    
    parser.add_argument('--pretrain_ratio', type=float, default=0.8, help="Ratio for pretraining (default: 0.8).")
    parser.add_argument('--pretrain_lr', type=float, default=2e-2, help="Learning rate for pretraining (default: 2e-2).")
    parser.add_argument('--pretrain_mask_type', type=str, default="entmax", help="Mask type for pretraining (default: entmax).")
    parser.add_argument('--pretrain_verbose', type=int, default=10, help="Verbosity level for pretraining (default: 10).")

    parser.add_argument('--n_d', type=int, default=64, help="Number of decision layer features (default: 64).")
    parser.add_argument('--n_a', type=int, default=64, help="Number of attention layer features (default: 64).")
    parser.add_argument('--n_steps', type=int, default=5, help="Number of decision steps (default: 5).")
    parser.add_argument('--gamma', type=float, default=1.5, help="Gamma parameter for TabNet (default: 1.5).")
    parser.add_argument('--lambda_sparse', type=float, default=1e-4, help="Sparse regularization lambda (default: 1e-4).")
    
    parser.add_argument('--lr', type=float, default=2e-2, help="Learning rate (default: 2e-2).")
    parser.add_argument('--step_size', type=int, default=10, help="Step size for learning rate scheduler (default: 10).")
    parser.add_argument('--gamma_lr', type=float, default=0.9, help="Gamma for learning rate scheduler (default: 0.9).")
    parser.add_argument('--patience', type=int, default=30, help="Patience for early stopping (default: 30).")
    
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training (default: 128).")
    parser.add_argument('--virtual_batch_size', type=int, default=256, help="Virtual batch size for TabNet (default: 256).")
    
    parser.add_argument('--verbose', type=int, default=10, help="Verbosity level (default: 10).")
    
    args = parser.parse_args()
    main(args)