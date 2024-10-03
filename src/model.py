import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.metrics import accuracy_score

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

    def save_model(self, path):
        self.model.save_model(path)

    def load_model(self, path):
        self.model.load_model(path)