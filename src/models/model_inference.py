import os
import pickle
import zipfile
from sklearn.metrics import accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier
from src.data.preprocess import preprocess_data
from glob import glob

class TabNetInference:
    def __init__(self, checkpoint_archive, task_number):
        self.checkpoint_archive = checkpoint_archive
        self.extracted_checkpoint_dir = os.path.splitext(os.path.basename(checkpoint_archive))[0]
        self.task_number = task_number
        self.model = None
        self.decode_labels = None

        self._ensure_checkpoint_exists()
        self._extract_checkpoint()
        self._load_model_and_decode_dict()

    def _ensure_checkpoint_exists(self):
        if not os.path.exists(self.checkpoint_archive):
            raise FileNotFoundError(f"Trained model archive not found at {self.checkpoint_archive}")

    def _extract_checkpoint(self):
        if not os.path.exists(self.extracted_checkpoint_dir):
            os.makedirs(self.extracted_checkpoint_dir, exist_ok=True)
        with zipfile.ZipFile(self.checkpoint_archive, 'r') as archive:
            archive.extractall(self.extracted_checkpoint_dir)
        print(f"Checkpoint extracted to {self.extracted_checkpoint_dir}")

    def _load_model_and_decode_dict(self):
        model_path = os.path.join(self.extracted_checkpoint_dir, f"trained_model_task{self.task_number}_model.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.model = TabNetClassifier()
        self.model.load_model(model_path)

        decode_dict_path = os.path.join(self.extracted_checkpoint_dir, f"trained_model_task{self.task_number}_decode.pkl")
        if not os.path.exists(decode_dict_path):
            raise FileNotFoundError(f"Decode dictionary not found at {decode_dict_path}")

        with open(decode_dict_path, 'rb') as f:
            self.decode_labels = pickle.load(f)
        self._cleanup()
    
    def _cleanup(self):
        """Remove the extracted checkpoint directory."""
        if os.path.exists(self.extracted_checkpoint_dir):
            for root, dirs, files in os.walk(self.extracted_checkpoint_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(self.extracted_checkpoint_dir)
            # print(f"Removed extracted checkpoint directory: {self.extracted_checkpoint_dir}")

    def load_images(self, images_folder):
        image_paths = glob(os.path.join(images_folder, "**/*.jpg"), recursive=True)
        if not image_paths:
            raise FileNotFoundError(f"No images found in the folder {images_folder}")

        X, y = preprocess_data(image_paths, self.task_number)
        return X, y

    def evaluate(self, X_test, y_test):
        y_pred_indices = self.model.predict(X_test)
        y_pred = [self.decode_labels[idx] for idx in y_pred_indices]
        y_true = [self.decode_labels[true_idx] for true_idx in y_test]

        accuracy = accuracy_score(y_true, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        return y_true, y_pred

    def predict(self, image_paths):
        """Predict labels for a list of image paths."""
        if not image_paths:
            raise ValueError("No image paths provided for prediction.")

        X, _ = preprocess_data(image_paths, self.task_number, save_data=False, process_labels=False)
        y_pred_indices = self.model.predict(X)
        y_pred = [self.decode_labels[idx] for idx in y_pred_indices]

        return y_pred