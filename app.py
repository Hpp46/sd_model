import gradio as gr
from main import generate_image
import requests

def translate(text, source_language, target_language):
    url = "https://libretranslate.de/translate"
    payload = {
        "q": text,
        "source": source_language,
        "target": target_language,
        "format": "text"
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["translatedText"]
    else:
        return text

def generate_and_translate(prompt, negative_prompt, guidance_scale, num_inference_steps, language):
    # 根据选择的语言决定是否进行翻译
    if language == "中文":
        translated_prompt = translate(prompt, "zh", "en")
        translated_negative_prompt = translate(negative_prompt, "zh", "en")
    else:
        translated_prompt = prompt
        translated_negative_prompt = negative_prompt

    image = generate_image(translated_prompt, translated_negative_prompt, guidance_scale, num_inference_steps)
    return image

with gr.Blocks() as demo:
    gr.Markdown("## Stable Diffusion Image Generator")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt（提示词）")
            negative_prompt = gr.Textbox(label="Negative Prompt（负面提示词）")
            guidance_scale = gr.Slider(1, 20, label="Guidance Scale（引导规模）", value=8)
            num_inference_steps = gr.Slider(1, 50, label="Number of Inference Steps（推理步数）", value=30)
            language = gr.Dropdown(["English", "中文"], label="Language（选择你要输入提示词的语言类型）", value="English")
            generate_button = gr.Button("Generate")

        with gr.Column():
            output_image = gr.Image()

    generate_button.click(
        generate_and_translate,
        inputs=[prompt, negative_prompt, guidance_scale, num_inference_steps, language],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch(share=True)
