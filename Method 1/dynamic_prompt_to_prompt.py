from PIL import Image
import torch
from prompt2prompt import MyLDMPipeline, MySharedAttentionSwapper, unet_inject_attention_modules, create_image_grid
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-7B-hf").to("cuda:1")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-hf")

story = input("Enter your storyline: ")

prefix_prompt = f"""
Generate a short story based on the given storyline, broken down into individual prompts. Each prompt should:
1. Focus on a specific action or moment in the story.
2. Maintain the consistency of the main character(s) and the setting.
3. Ensure that the length of each prompt is approximately the same (equal word count).
4. Follow the order of events logically, preserving the flow of the story.

The given storyline is:
"{story}"

Now, generate the story in consistent prompts:
"""

inputs = tokenizer(prefix_prompt, return_tensors="pt")
output = model.generate(
    **inputs,
    max_length=20,
    num_return_sequences=1,
    temperature=0.7
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

prompts_from_olmo = [
    sentence.strip()
    for sentence in generated_text.split("\n")
    if sentence.strip() and not sentence.startswith("Generate")
]

print("\nGenerated Prompts:")
for i, prompt in enumerate(prompts_from_olmo, 1):
    print(f"{i}: {prompt}")

# The prompts_from_olmo prompts should be similar to prompts which are shown below:
prompts = [
    "A young girl with long black hair walks calmly through a blooming garden in the morning light.",
    "The young girl notices a bed of roses nearby, their petals glistening with dew under the sunlight.",
    "The young girl looks up, seeing daisies and tulips swaying together in harmony with the gentle breeze.",
    "The young girl walks over to the daisies, adding them to her growing collection of hand-picked flowers.",
    "The young girl pauses by a lilac bush, smiling as its sweet fragrance fills the morning air."

]

for i in prompts:
    print(len(i.split(" ")))
pipe = MyLDMPipeline(num_inference_steps=20, guidance_scale=7.5)  # Increase steps for more gradual changes

attention_swapper = MySharedAttentionSwapper(
    prompts=prompts,
    tokenizer=pipe.tokenizer,
    prop_steps_cross=0.60,
    prop_steps_self=0.40
)

generated_images = []
base_image = None 

for i, prompt in enumerate(prompts):
    print(f"Generating image for prompt: {prompt}")
    image = pipe.generate_image_from_text(prompts[:i+1], attention_swapper)     
    print(f"Generated image {i+1} with {len(image)} elements")
    generated_images.append(image)
    image_path = f"output_image_{i+1}.png"
    image[i].save(image_path)
    print(f"Saved image: {image_path}")
    base_image = image[0]

story_images = [pipe.generate_image_from_text([p], attention_swapper)[0] for p in prompts]
grid_image = create_image_grid(story_images)
grid_image_path = "story_grid.png"
grid_image.save(grid_image_path)
print(f"Saved story grid: {grid_image_path}")
