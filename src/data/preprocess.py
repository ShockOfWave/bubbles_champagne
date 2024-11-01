import os
import pickle
import numpy as np
from tqdm import tqdm
from src.models.clip_inference import CLIPInference
from src.utils.config import converter


def extract_labels_from_path(path, label_index):
    """
    Extract a label from a given path.

    Args:
        path (str): A path to a file or directory.
        label_index (int): An index of the label to be extracted.

    Returns:
        str: The extracted label.

    Raises:
        ValueError: If the label index is out of range.
    """
    label_part = path.split('/')[-2]
    labels = label_part.split('_')
    
    if label_index >= len(labels):
        raise ValueError(f"Incorrect index label: {label_index}. In path {path} only {len(labels)} labels.")
    
    return labels[label_index]

def preprocess_data(paths, task_number, data_folder="data"):
    """
    Preprocesses data by extracting embeddings and labels from image paths.

    Args:
        paths (list of str): A list of file paths to the images to be processed.
        task_number (int): The task number used to determine the label index.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: An array of extracted embeddings from the images.
            - np.ndarray: An array of numerical labels corresponding to the images.
    """
    label_index = task_number - 1

    X = []
    y = []

    clip_model = CLIPInference()

    for path in tqdm(paths, desc="Processing images"):
        embeddings = clip_model.extract_embeddings(path)
        X.append(embeddings)
        
        label = extract_labels_from_path(path, label_index)
        y.append(label)
    
    y = [converter[x] for x in y]

    X = np.array(X)
    y = np.array(y)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    pickle.dump([X, y], open(os.path.join(data_folder, f"data_task{task_number}.pkl"), "wb"))
    return X, y
