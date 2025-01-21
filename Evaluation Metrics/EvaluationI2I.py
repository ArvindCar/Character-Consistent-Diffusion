import torch
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import os

model_ID = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_ID)

preprocess = CLIPImageProcessor.from_pretrained(model_ID)

def load_images(image_folder):
    images = []
    for i in range(1, 6):  
        filename = f'image_{i}.png'
        img_path = os.path.join(image_folder, filename)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB") 
            img = img.resize((512, 512))  
            images.append(img) 
        else:
            print(f"Warning: {filename} not found in the specified folder.")
            images.append(None)  
    return images

image_folder = ''
images = load_images(image_folder)

similarity_scores = []

for i in range(len(images) - 1):
    if images[i] is not None and images[i + 1] is not None:
        image_a = preprocess(images[i], return_tensors="pt")["pixel_values"]
        image_b = preprocess(images[i + 1], return_tensors="pt")["pixel_values"]

        with torch.no_grad():
            embedding_a = model.get_image_features(image_a)
            embedding_b = model.get_image_features(image_b)

        similarity_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)
        similarity_scores.append(similarity_score.item())

if similarity_scores:
    mean_similarity_score = sum(similarity_scores) / len(similarity_scores)
    print('Mean similarity score:', mean_similarity_score)
else:
    print('No valid similarity scores to compute mean.')