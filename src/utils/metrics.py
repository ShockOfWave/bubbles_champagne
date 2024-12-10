import torch
from pytorch_tabnet.metrics import Metric

class cross_entropy_reporting(Metric):
    def __init__(self):
        self._name = "cross_entropy" 
        self._maximize = False
    def __call__(self, y_true, y_score):
        return float(torch.nn.functional.cross_entropy(torch.tensor(y_score), torch.tensor(y_true)))