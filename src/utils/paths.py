import os
from pathlib import Path

def get_project_path() -> Path:
    return Path(__file__).parent.parent.parent


def get_segmentation_model_path() -> Path:
    return os.path.join(get_project_path(), "models", "segmentation_model.pt")


def get_checkpoint_path() -> Path:
    return os.path.join(get_project_path(), "checkpoints")