from src.utils import task_types
from src.models.main_model import VideoClassifier
from src.models.clip_inference import CLIPInference
from src.models.segmentation_inference import SegmentationInference


class MainModelInference:
    def __init__(self, model_path, task_type):
        self.task_number = self.get_task_type(task_type)
        self.model = VideoClassifier()
        self.model.load_model(model_path, self.task_number)
        self.clip_inference = CLIPInference()
        self.segmentation_inference = SegmentationInference()
        
    def predict(self, X):
        seg_pred = self.segmentation_inference.predict(X)
        if seg_pred.masks and hasattr(seg_pred.masks, 'xy'):
            X = self.clip_inference.extract_embeddings(X)
            pred = self.model.predict([X])
            predicted_label = self.model.decode[pred[0]]
            return predicted_label
        else:
            return False
    
    def predict_proba(self, X):
        X = self.clip_inference.extract_embeddings(X)
        pred = self.model.predict_proba(X)
        return pred

    def get_task_type(self, task_type):
        return task_types[task_type]