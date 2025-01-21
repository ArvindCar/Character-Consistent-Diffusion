from diffusers import StableDiffusionPipeline, StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import numpy as np
import cv2
from PIL import Image

story_prompts = [
    "A photo of a 50 years old man with curly hair.",
    "A photo of a 50 years old man with curly hair in the park.",
    "A photo of a 50 years old man with curly hair reading a book.",
    "A photo of a 50 years old man with curly hair at the beach.",
    "A photo of a 50 years old man with curly hair holding an avocado.",
]

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", controlnet=controlnet, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

text2img_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
text2img_pipe.enable_model_cpu_offload()

generator = torch.manual_seed(0)
first_image = text2img_pipe(story_prompts[0], num_inference_steps=20, generator=generator).images[0]
first_image.save("image_1.png")

previous_image = first_image
for i in range(1, len(story_prompts)):
    print(f"Generating image {i + 1}...")
    np_image = np.array(previous_image)
    np_image = cv2.Canny(np_image, 100, 200)
    np_image = np_image[:, :, None]
    np_image = np.concatenate([np_image, np_image, np_image], axis=2)
    canny_image = Image.fromarray(np_image)

    next_image = pipe(
        story_prompts[i],
        num_inference_steps=20,
        generator=generator,
        image=previous_image if previous_image is not None else first_image,
        control_image=canny_image,
        controlnet_conditioning_scale=0.3
    ).images[0]
    next_image.save(f"image_{i + 1}.png")
    previous_image = next_image

print("All images generated and saved successfully.")
