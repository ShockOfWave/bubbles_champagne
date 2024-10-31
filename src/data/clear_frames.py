import os
from glob import glob
from tqdm import tqdm
import cv2
from src.models.segmentation_inference import SegmentationInference


def process_images_in_directory(directory_path, segmentation_model):
    """
    Processes all images in a given directory and its subdirectories using a segmentation model.
    
    This function searches for image files with extensions .jpg, .jpeg, and .png within the specified directory
    and its subdirectories, then applies the provided segmentation model to each image. If an image is determined
    to be empty or not useful by the `process_image` function, it is removed from the filesystem.

    Args:
        directory_path (str): The path to the directory containing images to be processed.
        segmentation_model: An instance of a segmentation model used to process the images.

    Returns:
        None
    """
    image_paths = glob(os.path.join(directory_path, "**", "*.jpg"), recursive=True) + \
                  glob(os.path.join(directory_path, "**", "*.jpeg"), recursive=True) + \
                  glob(os.path.join(directory_path, "**", "*.png"), recursive=True)
    
    for image_path in tqdm(image_paths, desc="Deleting empty images"):
        if not process_image(image_path, segmentation_model):
            os.remove(image_path)


def process_image(image_path, segmentation_model):
    """
    Processes an image using a segmentation model to determine its usefulness.

    This function reads an image from the specified path and applies a segmentation model
    to it. If the segmentation results contain masks with coordinates, the image is considered
    useful.

    Args:
        image_path (str): The path to the image file to be processed.
        segmentation_model: An instance of a segmentation model used to process the image.

    Returns:
        bool: True if the image is considered useful based on the segmentation results, False otherwise.
    """
    image = cv2.imread(image_path)
    results = segmentation_model.predict(image)
    
    if results.masks and hasattr(results.masks, 'xy'):
        return True
    
    return False


def process_project_directories(project_root):
    """
    Processes image directories for a given project root using a segmentation model.

    This function iterates through the 'train', 'val', and 'test' directories within the specified
    project root directory. For each directory that exists, it processes all images using a segmentation
    model to determine their usefulness. If the directory does not exist, a message is printed.

    Args:
        project_root (str): The path to the root directory of the project, which contains the 'train',
                            'val', and 'test' directories.

    Returns:
        None
    """
    segmentation_model = SegmentationInference()
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(project_root, split)
        if os.path.exists(split_path):
            process_images_in_directory(split_path, segmentation_model)
        else:
            print(f"Path {split_path} not found.")
            

if __name__ == "__main__":
    process_project_directories()
