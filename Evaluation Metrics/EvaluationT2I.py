import torch
from torchmetrics.multimodal.clip_score import CLIPScore
from functools import partial
import os
import numpy as np
from PIL import Image

def load_images(image_folder):
    images = []
    for i in range(1, 6):  
        folder_name = f'timestep_{i}'  
        img_path = os.path.join(image_folder, folder_name, 'image.png') 
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img = img.resize((512, 512))  
            img_array = np.array(img)  
            images.append(img_array)  
        else:
            print(f"Warning: img.png not found in folder {folder_name}.")
    
    if images:  
        return np.stack(images)  
    else:
        raise ValueError("No images were loaded. Please check the input folders.")

def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)


image_folder = ''
prompts = [
    "A cute albino porcupine sits indoors under the cozy warm lighting.",
    "A cute albino porcupine walks through a park with green grass.",
    "A cute albino porcupine quietly reads a book in the park",
    "A cute albino porcupine relaxes on the beach near gentle waves.",
    "A cute albino porcupine holds an avocado with both tiny paws."

]

images = load_images(image_folder)


clip_score_fn = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32")
sd_clip_score = calculate_clip_score(images, prompts)
print(f"CLIP score: {sd_clip_score}")
