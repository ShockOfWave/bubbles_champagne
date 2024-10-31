import argparse
from glob import glob
from src.data.clear_frames import process_project_directories
from src.data.crop_frames import process_videos
from src.data.split_videos import split_videos
from src.models.main_model import VideoClassifier
from src.data.preprocess import preprocess_data
from src.utils import decode, project_path, task_types
import os
import optuna


parser = argparse.ArgumentParser(description="Video classification training")

parser.add_argument('--root_dir', type=str, required=True, help="Path to the train data directory")
args = parser.parse_args()

if not os.path.exists("data_split/"):
    split_videos(root_dir=args.root_dir, output_dir="data_split")
else:
    print("Data split already exists. Skipping data split...")

if not os.path.exists("frames/"):
    process_videos(root_dir="data_split", output_dir="frames")
else:
    print("Frames already extracted. Skipping frame extraction...")

print('Clearing frames...')
process_project_directories(project_root='frames')
print('Frames cleared.')

train_paths = glob("frames/train/*/*.jpg")
val_paths = glob("frames/val/*/*.jpg")
test_paths = glob("frames/test/*/*.jpg")

    
def objective(trial):
    n_d = trial.suggest_int('n_d', 32, 128)
    n_a = trial.suggest_int('n_a', 32, 128)
    n_steps = trial.suggest_int('n_steps', 5, 10)
    gamma = trial.suggest_float('gamma', 1.0, 2.0)
    lambda_sparse = trial.suggest_float('lambda_sparse', 1e-4, 1e-1)
    step_size = trial.suggest_int('step_size', 5, 20)
    gamma_lr = trial.suggest_float('gamma_lr', 0.9, 0.99)
    batch_size = trial.suggest_int('batch_size', 128, 256)
    virtual_batch_size = trial.suggest_int('virtual_batch_size', 128, 256)
    patience = 50
    pretrain_ratio = trial.suggest_float('pretrain_ratio', 0.5, 0.9)
    max_epochs = 1000

    model = VideoClassifier(
        n_d=n_d, 
        n_a=n_a, 
        n_steps=n_steps, 
        gamma=gamma, 
        lambda_sparse=lambda_sparse, 
        lr=1e-5, 
        step_size=step_size, 
        gamma_lr=gamma_lr, 
        batch_size=batch_size, 
        virtual_batch_size=virtual_batch_size,
        max_epochs=max_epochs
    )
    
    model.decode = decode[task_number - 1]

    model.pretrain(X_train, X_val, pretrain_ratio=pretrain_ratio)

    model.train(
        X_train, y_train, X_val, y_val,
        patience=patience
    )
    
    accuracy = model.evaluate(X_val, y_val)
    return accuracy


for task in ["champagne_type", "container_type", "time"]:
    
    task_number = task_types[task]
    
    print("Preprocessing training data...")
    X_train, y_train = preprocess_data(train_paths, task_number)

    print("Preprocessing validation data...")
    X_val, y_val = preprocess_data(val_paths, task_number)

    print("Preprocessing test data...")
    X_test, y_test = preprocess_data(test_paths, task_number)
    
    study = optuna.create_study(
                study_name=f'bubble_champagne_{task}',
                direction='maximize',
                pruner=optuna.pruners.MedianPruner(),
                load_if_exists=True,
                storage=f'sqlite:///{project_path}/optuna_models_optimization.db'
            )
    
    study.optimize(objective, n_trials=100)
