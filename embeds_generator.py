import cv2
import torch 
import numpy as np 
from transformers import CLIPProcessor, CLIPModel

device = torch.device("cuda" if torch.cuda.is_availible() else "cpu")

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def extract_embeddings(img_path):
    image = cv2.imread(img_path)
    inputs = processor(images=image, return_tensors="pt", padding=True).to("cuda:0")
    with torch.no_grad():
        embed = model.get_image_features(**inputs).cpu().squeeze(0).numpy()
    return embed