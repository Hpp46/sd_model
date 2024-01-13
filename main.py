# fu.py
# ... [之前的所有代码] ...
# Import necessary libraries
import torch
import requests
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionDepth2ImgPipeline
)
from transformers import pipeline

# Set device
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)


# Function to download images
def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

# Download images for inpainting example
img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

def generate_image(prompt, negative_prompt, guidance_scale=8, num_inference_steps=30):

    # Encode the prompt
    text_embeddings = pipe._encode_prompt(prompt, device, 1, True, negative_prompt)

    # Create our random starting point
    generator = torch.Generator(device=device).manual_seed(0)
    latents = torch.randn((1, 4, 64, 64), device=device, generator=generator)
    latents *= pipe.scheduler.init_noise_sigma

    # Prepare the scheduler
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Loop through the sampling timesteps
    for i, t in enumerate(pipe.scheduler.timesteps):
        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)

        # Apply any scaling required by the scheduler
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual with the UNet
        with torch.no_grad():
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Decode the resulting latents into an image
    with torch.no_grad():
        image = pipe.decode_latents(latents.detach())

    # Convert to PIL image and return
    result_image = pipe.numpy_to_pil(image)[0]

    return result_image
