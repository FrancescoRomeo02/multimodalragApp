import base64
from io import BytesIO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import logging
import cv2
import pytesseract
import os
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')

def get_caption(base64_str):
    model_path = "Salesforce/blip-image-captioning-base"
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache/huggingface")

    logging.info("Loading BLIP model.")
    processor = BlipProcessor.from_pretrained(model_path, cache_dir=cache_dir)
    model = BlipForConditionalGeneration.from_pretrained(model_path, cache_dir=cache_dir)
    logging.info("Model loaded successfully.")

    # Decodifica base64 e converte in immagine PIL
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data)).convert("RGB")

    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=100)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption

def get_image_text(base64_str: str):
    # Decodifica base64 e converte in immagine OpenCV
    image_data = base64.b64decode(base64_str)
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return pytesseract.image_to_string(image_rgb)