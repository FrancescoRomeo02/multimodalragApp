# file: app/utils/image_info.py

import base64
from io import BytesIO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import logging
import cv2
import pytesseract
import os
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_detected_objects(base64_str: str) -> list:
    """
    Rileva oggetti in un'immagine usando YOLO.
    
    Args:
        base64_str: Stringa base64 dell'immagine
        
    Returns:
        Una lista di oggetti rilevati con il loro nome e confidenza.
    """
    try:
        model = YOLO("yolov8n.pt") # Carica un modello pre-addestrato
        
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        results = model(image)
        
        detected_objects = []
        for result in results:
            for box in result.boxes:
                label = model.names[int(box.cls)]
                confidence = float(box.conf)
                detected_objects.append(f"{label} (confidenza: {confidence:.2f})")
                
        return detected_objects
    except Exception as e:
        logger.error(f"Errore nel rilevamento oggetti: {e}")
        return []


def get_caption(base64_str: str) -> str:
    """
    Genera una descrizione testuale di un'immagine usando BLIP model.
    
    Args:
        base64_str: Stringa base64 dell'immagine
        
    Returns:
        Descrizione testuale dell'immagine
    """
    try:
        model_path = "Salesforce/blip-image-captioning-base"
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache/huggingface")

        logger.info("Caricamento modello BLIP...")
        processor = BlipProcessor.from_pretrained(model_path, cache_dir=cache_dir)
        model = BlipForConditionalGeneration.from_pretrained(model_path, cache_dir=cache_dir)

        # Decodifica e conversione immagine
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Generazione caption
        inputs = processor(image, return_tensors="pt") # type: ignore
        out = model.generate(**inputs, max_new_tokens=100) # type: ignore
        caption = processor.decode(out[0], skip_special_tokens=True) # type: ignore

        return caption

    except Exception as e:
        logger.error(f"Errore generazione caption: {e}")
        return "Immagine non descrivibile"

def get_image_text(base64_str: str) -> str:
    """
    Estrae testo da un'immagine usando OCR (Tesseract).
    
    Args:
        base64_str: Stringa base64 dell'immagine
        
    Returns:
        Testo estratto dall'immagine
    """
    try:
        # Decodifica e conversione immagine
        image_data = base64.b64decode(base64_str)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Impossibile decodificare immagine")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return pytesseract.image_to_string(image_rgb)
        
    except Exception as e:
        logger.error(f"Errore OCR: {e}")
        return ""

def get_comprehensive_image_info(base64_str: str) -> dict:
    """
    Estrae un set completo di informazioni da un'immagine.
    
    Args:
        base64_str: Stringa base64 dell'immagine
        
    Returns:
        Un dizionario contenente tutte le informazioni estratte.
    """
    info = {
        "caption": get_caption(base64_str),
        "ocr_text": get_image_text(base64_str),
        "detected_objects": get_detected_objects(base64_str),
    }
    return info
