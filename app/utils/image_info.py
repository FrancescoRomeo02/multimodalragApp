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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=100)
        caption = processor.decode(out[0], skip_special_tokens=True)

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

def get_image_description(base64_str: str) -> str:
    """
    Funzione combinata che restituisce sia caption che testo estratto.
    Mantenuta per compatibilità con il codice esistente.
    
    Args:
        base64_str: Stringa base64 dell'immagine
        
    Returns:
        Stringa combinata con caption e testo estratto
    """
    caption = get_caption(base64_str)
    text = get_image_text(base64_str)
    
    # Combina i risultati eliminando spazi vuoti
    parts = [part for part in [caption, text] if part.strip()]
    return "\n".join(parts)

# Esempio di utilizzo
if __name__ == "__main__":
    # Test con un'immagine locale
    test_image_path = "test_image.png"
    if os.path.exists(test_image_path):
        with open(test_image_path, "rb") as image_file:
            test_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        
        print("Caption:", get_caption(test_base64))
        print("Testo estratto:", get_image_text(test_base64))
        print("Descrizione completa:", get_image_description(test_base64))
    else:
        print(f"File {test_image_path} non trovato per il test")