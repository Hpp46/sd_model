# ui.py
import gradio as gr
from main import generate_image

def generate(prompt, negative_prompt, guidance_scale, num_inference_steps):
    image = generate_image(prompt, negative_prompt, guidance_scale, num_inference_steps)
    return image

iface = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Textbox(label="Negative Prompt"),
        gr.Slider(1, 20, label="Guidance Scale", value=8),
        gr.Slider(1, 50, label="Number of Inference Steps", value=30)
    ],
    outputs="image",
    title="Stable Diffusion Image Generator",
    description="Generate images using the Stable Diffusion model"
)


if __name__ == "__main__":
    iface.launch(share=True)
