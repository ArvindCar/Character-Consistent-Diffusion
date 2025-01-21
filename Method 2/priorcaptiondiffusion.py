import os
import json
import spacy
from diffusers import StableDiffusionPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nlp = spacy.load("en_core_web_sm")

def extract_dynamic_elements(caption):

    doc = nlp(caption)
    adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]

    return {
        "adjectives": adjectives,
        "nouns": nouns,
        "noun_phrases": noun_phrases
    }

def construct_next_prompt_from_context(elements, base_prompt):

    adjectives = ", ".join(elements.get("adjectives", []))
    nouns = ", ".join(elements.get("nouns", []))
    noun_phrases = ", ".join(elements.get("noun_phrases", []))

    context = f"The scene includes {adjectives} features, focusing on {noun_phrases or nouns}."
    return f"{base_prompt}. {context}"

def create_timestep_folder(step):

    folder_name = f"timestep_{step + 1}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

def save_timestep_data(folder_path, image, caption, elements):

    image.save(os.path.join(folder_path, "image.png"))

    with open(os.path.join(folder_path, "caption.txt"), "w") as f:
        f.write(caption)

    with open(os.path.join(folder_path, "elements.json"), "w") as f:
        json.dump(elements, f, indent=4)

def combine_images_sequentially(folder_paths, output_path="combined.jpg"):

    images = [Image.open(os.path.join(folder, "image.png")) for folder in folder_paths]
    widths, heights = zip(*(img.size for img in images))

    total_width = sum(widths)
    max_height = max(heights)

    combined_image = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for img in images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    combined_image.save(output_path)
    print(f"Combined image saved at: {output_path}")


pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

storyline_prompts = [
    # "A knight embarking on a quest in a mystical forest",
    # "A dark cave with glowing crystals",
    # "A majestic castle in the distance",
    # "A fierce dragon guarding a treasure",
    # "A celebration in the village after the quest"

    # "A cute albino porcupine sits indoors under the cozy warm lighting.",
    # "A cute albino porcupine walks through a park with green grass.",
    # "A cute albino porcupine quietly reads a book in the park",
    # "A cute albino porcupine relaxes on the beach near gentle waves.",
    # "A cute albino porcupine holds an avocado with both tiny paws."

    # "A photo of a 50 years old man with curly hair sits indoors under the cozy warm lighting.",
    # "A photo of a 50 years old man with curly hair walks through a park with green grass.",
    # "A photo of a 50 years old man with curly hair quietly reads a book in the park",
    # "A photo of a 50 years old man with curly hair relaxes on the beach near gentle waves.",
    # "A photo of a 50 years old man with curly hair holds an avocado with both tiny paws."

    "A photo of a 50 years old man with curly hair.",
    "A photo of a 50 years old man with curly hair in the park.",
    "A photo of a 50 years old man with curly hair reading a book.",
    "A photo of a 50 years old man with curly hair at the beach.",
    "A photo of a 50 years old man with curly hair holding an avocado."
]

previous_elements = {}
folder_paths = []
for step, base_prompt in enumerate(storyline_prompts):

    if step > 0:
        base_prompt = construct_next_prompt_from_context(previous_elements, base_prompt)

    print(f"Timestep {step + 1}: {base_prompt}")

    image = pipe(base_prompt).images[0]
    folder_path = create_timestep_folder(step)
    folder_paths.append(folder_path) 

    inputs = processor(image, return_tensors="pt").to(device)
    caption_ids = model.generate(**inputs, max_length=150, num_beams=7, no_repeat_ngram_size=3, length_penalty=1.2)
    caption = processor.decode(caption_ids[0], skip_special_tokens=True)

    previous_elements = extract_dynamic_elements(caption)

    save_timestep_data(folder_path, image, caption, previous_elements)

combine_images_sequentially(folder_paths, output_path="all_timesteps_combined.jpg")
