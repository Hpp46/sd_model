# try1.py
import torch
import requests
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
from diffusers import StableDiffusionPipeline


def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


def main():
    # 下载图片
    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
    init_image = download_image(img_url).resize((512, 512))
    mask_image = download_image(mask_url).resize((512, 512))

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


    model_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

    generator = torch.Generator(device=device).manual_seed(42)


    pipe_output = pipe(
        prompt="",
        negative_prompt="Oversaturated, blurry, low quality",
        height=480, width=640,
        guidance_scale=8,
        num_inference_steps=35,
        generator=generator
    )


    plt.imshow(pipe_output.images[0])
    plt.show()

if __name__ == "__main__":
    main()
