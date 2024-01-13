import gradio as gr
from main import generate_image
import requests


# 使用 LibreTranslate 进行翻译
def translate(text, source_language="zh", target_language="en"):
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
        return text  # 如果请求失败，返回原文本


def generate_and_translate(prompt, negative_prompt, guidance_scale, num_inference_steps):
    # 将prompt和negative prompt翻译成英文
    translated_prompt = translate(prompt)
    translated_negative_prompt = translate(negative_prompt)

    # 使用翻译后的文本生成图像
    image = generate_image(translated_prompt, translated_negative_prompt, guidance_scale, num_inference_steps)
    return image


iface = gr.Interface(
    fn=generate_and_translate,
    inputs=[
        gr.Textbox(label="提示语（中文）"),
        gr.Textbox(label="否定提示语（中文）"),
        gr.Slider(1, 20, label="引导规模", value=8),
        gr.Slider(1, 50, label="推理步数", value=30)
    ],
    outputs="image",
    title="Stable Diffusion Image Generator",
    description="使用中文提示生成图像"
)

if __name__ == "__main__":
    iface.launch(share=True)
