import cv2
import torch
import numpy as np
from pathlib import Path
from transformers import AutoProcessor, AutoModel


class CLIPInference:
    def __init__(self):
        """
        Initialize the CLIPInference class.

        This method sets up the CLIP model and processor.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(self.device)
        self.processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

    def extract_embeddings(self, img: str | np.ndarray):
        """
        Extract an embedding from an image.

        Parameters
        ----------
        img_path : str
            Path to the image to extract the embedding from.

        Returns
        -------
        embed : np.ndarray
            The embedding extracted from the image.
        """
        if type(img) is str or type(img) is Path:
            img = cv2.imread(img)
        with torch.no_grad():
            inputs = self.processor(images=img, return_tensors="pt", padding=True).to(self.device)
            embed = self.model.get_image_features(**inputs).cpu().squeeze(0).numpy()
        return embed
    