import torch
from PIL import Image
from gradio_client import Client, handle_file
import google.generativeai as genai

class GeminiCaption:
    def __init__(self, device,gemini_api_key):
        self.device = device
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")

        
    def inference(self, image_path):
        img = Image.open(image_path)

        text = f'Imagine you are A remote sensing agent. Give caption of the this image in details'
        result = self.model.generate_content([text, img])

        return result.text
    


