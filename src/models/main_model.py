import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.metrics import accuracy_score
import pickle
import zipfile
import os
from catboost import CatBoostClassifier

class VideoClassifier:
    def __init__(self, n_d=64, n_a=64, n_steps=5, gamma=1.5, lambda_sparse=1e-4, lr=2e-2, step_size=10, gamma_lr=0.9, batch_size=128, virtual_batch_size=256, verbose=10, max_epochs=1000):
        """
        Initializes the VideoClassifier with specified hyperparameters for the TabNet model and pretrainer.

        Args:
            n_d (int): Dimensionality of the decision prediction layer. Default is 64.
            n_a (int): Dimensionality of the attention embedding for each mask. Default is 64.
            n_steps (int): Number of steps in the architecture. Default is 5.
            gamma (float): Relaxation parameter in the range [1, 2] for the TabNet model. Default is 1.5.
            lambda_sparse (float): Coefficient for feature sparsity regularization. Default is 1e-4.
            lr (float): Learning rate for the optimizer. Default is 2e-2.
            step_size (int): Period of learning rate decay for the scheduler. Default is 10.
            gamma_lr (float): Multiplicative factor of learning rate decay. Default is 0.9.
            batch_size (int): Number of samples per batch. Default is 128.
            virtual_batch_size (int): Size of the mini-batch for ghost batch normalization. Default is 256.
            verbose (int): Verbosity level of the training process. Default is 10.

        Attributes:
            model (TabNetClassifier): The TabNet classifier model.
            pretrainer (TabNetPretrainer): The TabNet pretrainer model.
            batch_size (int): Number of samples per batch.
            virtual_batch_size (int): Size of the mini-batch for ghost batch normalization.
            __pretrained (bool): Indicator of whether the model has been pretrained.
            decode (dict or None): Dictionary for decoding labels for each task.
        """
        self.model = TabNetClassifier(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=lr),
            scheduler_params={"step_size":10, # how to use learning rate scheduler
                                "gamma":0.9},
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
        self.decode = None
        self.max_epochs = max_epochs

    def pretrain(self, X_train, X_val, pretrain_ratio=0.8):
        """
        Pretrains the TabNet model using unsupervised learning.

        Args:
            X_train (array-like): Training data for pretraining.
            X_val (array-like): Validation data for evaluating pretraining progress.
            pretrain_ratio (float, optional): Ratio of training data to use for pretraining. Default is 0.8.

        This function sets the model's pretrained status to True upon completion.
        """
        self.pretrainer.fit(
            X_train=X_train,
            eval_set=[X_val],
            pretraining_ratio=pretrain_ratio,
            max_epochs=self.max_epochs
        )
        self.__pretrained = True
        print("Pretraining completed!")

    def train(self, X_train, y_train, X_val, y_val, patience=30):
        """
        Trains the TabNet model using the provided training and validation datasets.

        Args:
            X_train (array-like): Training data features.
            y_train (array-like): Training data labels.
            X_val (array-like): Validation data features.
            y_val (array-like): Validation data labels.
            patience (int, optional): Number of epochs with no improvement on validation metric before stopping. Default is 30.

        If the model has not been pretrained, pretraining is performed before the main training process.
        """
        if not self.__pretrained:
            print("Model hasn't been pre-trained yet!")
            print("Pretraining...")
            self.pretrain(X_train, X_val, pretrain_ratio=0.8)
            
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=["balanced_accuracy", "accuracy"],
            patience=patience,
            from_unsupervised=self.pretrainer,  # Используем предобученную модель
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size,
            max_epochs=self.max_epochs
        )

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        with torch.no_grad():
            y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {acc:.4f}")
        return acc

    def save_model(self, path):
        """
        Saves the model, pretrainer and decode dictionary to a .zip archive.

        Args:
            path (str): Path to the output file.

        The model, pretrainer and decode dictionary are saved to three separate files:
        `<path>_model.zip`, `<path>_pretrainer.zip` and `<path>_decode.pkl`, respectively.
        These files are then added to a .zip archive at `<path>.zip`.

        The original files are removed after archiving.
        """
        model_save_path = f"{path}_model.zip"
        pretrainer_save_path = f"{path}_pretrainer.zip"
        decode_save_path = f"{path}_decode.pkl"

        self.model.save_model(model_save_path[:-4])
        self.pretrainer.save_model(pretrainer_save_path[:-4])
        
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
        Loads the model, pretrainer and decode dictionary from a .zip archive.

        Args:
            path (str): Path to the input file.
            task_number (int): Task number (1, 2, or 3).

        The .zip archive is expected to contain three files: `<path>_task{task_number}_model.zip`,
        `<path>_task{task_number}_pretrainer.zip` and `<path>_task{task_number}_decode.pkl`.

        The original files are removed after loading.
        """
        archive_path = os.path.join(path, f"trained_model_task{task_number}.zip")
        
        with zipfile.ZipFile(archive_path, 'r') as archive:
            archive.extractall(path=os.path.dirname(archive_path))
        
        model_load_path = os.path.join(path, f"trained_model_task{task_number}_model.zip")
        pretrainer_load_path = os.path.join(path, f"trained_model_task{task_number}_pretrainer.zip")
        decode_load_path = os.path.join(path, f"trained_model_task{task_number}_decode.pkl")
        model = TabNetClassifier()
        model.load_model(model_load_path) 
        self.model = model
        self.pretrainer.load_model(pretrainer_load_path)

        with open(decode_load_path, 'rb') as f:
            self.decode = pickle.load(f)

        os.remove(model_load_path)
        os.remove(pretrainer_load_path)
        os.remove(decode_load_path)

        self.__pretrained = True  
        print(f"Model, pretrainer, and decode dictionary loaded from {archive_path}")
