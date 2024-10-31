from src.utils.config import converter, decode, task_types
from src.utils.paths import get_project_path, get_checkpoint_path, get_segmentation_model_path

project_path = get_project_path()
checkpoint_path = get_checkpoint_path()
segmentation_model_path = get_segmentation_model_path()

__all__ = ["converter", "decode", "task_types", "project_path", "checkpoint_path", "segmentation_model_path"]
