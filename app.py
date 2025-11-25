import gradio as gr
import requests
from io import BytesIO

API_URL = "https://mlops-lab2-latest.onrender.com"

def predict_image(file_path):
    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path, f, "image/png")}
            response = requests.post(f"{API_URL}/predict", files=files, timeout=10)
            response.raise_for_status()
            data = response.json()
            return f"Prediction: {data.get('prediction')}"
    except requests.exceptions.HTTPError as e:
        return f"Error: {response.json().get('detail', str(e))}"

def resize_image(file_path, width, height):
    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path, f, "image/png")}
            data = {"width": width, "height": height}
            response = requests.post(f"{API_URL}/resize", files=files, data=data, timeout=10)
            response.raise_for_status()
            return BytesIO(response.content)
    except requests.exceptions.HTTPError as e:
        return f"Error: {response.json().get('detail', str(e))}"

def grayscale_image(file_path):
    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path, f, "image/png")}
            response = requests.post(f"{API_URL}/grayscale", files=files, timeout=10)
            response.raise_for_status()
            return BytesIO(response.content)
    except requests.exceptions.HTTPError as e:
        return f"Error: {response.json().get('detail', str(e))}"

with gr.Blocks() as demo:
    gr.Markdown("## MLOps Lab 2 - Image Processing API")

    with gr.Tab("Predict"):
        img_input = gr.Image(type="filepath", label="Upload image")
        predict_btn = gr.Button("Predict")
        predict_output = gr.Textbox(label="Prediction")
        predict_btn.click(predict_image, inputs=img_input, outputs=predict_output)

    with gr.Tab("Resize"):
        resize_img = gr.Image(type="filepath", label="Upload image")
        width_input = gr.Number(label="Width", value=256)
        height_input = gr.Number(label="Height", value=256)
        resize_btn = gr.Button("Resize")
        resize_output = gr.Image(label="Resized image")
        resize_btn.click(resize_image, inputs=[resize_img, width_input, height_input], outputs=resize_output)

    with gr.Tab("Grayscale"):
        gray_img = gr.Image(type="filepath", label="Upload image")
        gray_btn = gr.Button("Convert to Grayscale")
        gray_output = gr.Image(label="Grayscale image")
        gray_btn.click(grayscale_image, inputs=gray_img, outputs=gray_output)

if __name__ == "__main__":
    demo.launch()
