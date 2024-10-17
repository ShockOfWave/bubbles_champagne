import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.metrics import accuracy_score
import pickle
import zipfile
import os

class VideoClassifier:
    def __init__(self, n_d=64, n_a=64, n_steps=5, gamma=1.5, lambda_sparse=1e-4, lr=2e-2, step_size=10, gamma_lr=0.9, batch_size=128, virtual_batch_size=256, verbose=10):
        self.model = TabNetClassifier(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=lr),
            scheduler_params={"step_size": step_size, "gamma": gamma_lr},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            verbose=verbose
        )
        self.pretrainer = TabNetPretrainer(
            optimizer_fn=torch.optim.AdamW,
            optimizer_params=dict(lr=lr),
            mask_type='entmax',
            verbose=verbose
        )
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.__pretrained = False
        self.decode = None  # Словарь для декодирования меток для каждой задачи

    def pretrain(self, X_train, X_val, pretrain_ratio=0.8):
        self.pretrainer.fit(
            X_train=X_train,
            eval_set=[X_val],
            pretraining_ratio=pretrain_ratio,
        )
        self.__pretrained = True
        print("Pretraining completed!")

    def train(self, X_train, y_train, X_val, y_val, patience=30):
        if not self.__pretrained:
            print("Model hasn't been pre-trained yet!")
            print("Pretraining...")
            self.pretrain(X_train, X_val, pretrain_ratio=0.8)
            
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=["accuracy", "balanced_accuracy"],
            patience=patience,
            from_unsupervised=self.pretrainer,  # Используем предобученную модель
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size,
        )

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {acc:.4f}")
        return acc

    def save_model(self, path, task_number):
        """
        Сохраняет модель, предтренер и словарь decode в архив .zip
        """
        model_save_path = f"{path}_model.zip"
        pretrainer_save_path = f"{path}_pretrainer.zip"
        decode_save_path = f"{path}_decode.pkl"

        self.model.save_model(model_save_path)
        self.pretrainer.save_model(pretrainer_save_path)
        
        with open(decode_save_path, 'wb') as f:
            pickle.dump(self.decode, f)
        
        archive_path = f"{path}.zip"
        with zipfile.ZipFile(archive_path, 'w') as archive:
            archive.write(model_save_path, os.path.basename(model_save_path))
            archive.write(pretrainer_save_path, os.path.basename(pretrainer_save_path))
            archive.write(decode_save_path, os.path.basename(decode_save_path))

        os.remove(model_save_path)
        os.remove(pretrainer_save_path)
        os.remove(decode_save_path)

        print(f"Model, pretrainer, and decode dictionary saved to {archive_path}")

    def load_model(self, path, task_number):
        """
        Загружает модель, предтренер и словарь decode из архива .zip
        """
        archive_path = f"{path}_task{task_number}.zip"
        
        with zipfile.ZipFile(archive_path, 'r') as archive:
            archive.extractall(path=os.path.dirname(archive_path))
        
        model_load_path = f"{path}_model.zip"
        pretrainer_load_path = f"{path}_pretrainer.zip"
        decode_load_path = f"{path}_decode.pkl"

        self.model.load_model(model_load_path)
        self.pretrainer.load_model(pretrainer_load_path)

        with open(decode_load_path, 'rb') as f:
            self.decode = pickle.load(f)

        os.remove(model_load_path)
        os.remove(pretrainer_load_path)
        os.remove(decode_load_path)

        self.__pretrained = True  
        print(f"Model, pretrainer, and decode dictionary loaded from {archive_path}")
