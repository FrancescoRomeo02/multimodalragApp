# file: app/utils/image_info.py

import base64
from io import BytesIO
from PIL import Image
from groq import Groq
import logging
import cv2
import pytesseract
import numpy as np
from ultralytics import YOLO
from src.config import GROQ_API_KEY, IMG_DESC_MODEL_LG

logger = logging.getLogger(__name__)


def get_detected_objects(base64_str: str) -> list:
    """
    Detects objects in an image using YOLO.
    
    Args:
        base64_str: Base64 string of the image
        
    Returns:
        A list of detected objects with their name and confidence.
    """
    try:
        # Path relativo al modello YOLO
        import os
        from pathlib import Path
        
        # Ottieni il path del progetto (root)
        project_root = Path(__file__).parent.parent.parent
        model_path = project_root / "data" / "models" / "yolov8n.pt"
        
        model = YOLO(str(model_path))  # Carica il modello dal path corretto
        
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        results = model(image)
        
        detected_objects = []
        for result in results:
            for box in result.boxes:
                label = model.names[int(box.cls)]
                confidence = float(box.conf)
                detected_objects.append(f"{label}" if confidence > 0.5 else None)

        return [obj for obj in detected_objects if obj is not None]
    except Exception as e:
        logger.error(f"Error in object detection: {e}")
        return []

def get_caption(base64_str: str) -> str:
    """
    Generates a textual description of an image using Groq's vision model.
    
    Args:
        base64_str: Base64 string of the image
        
    Returns:
        Textual description of the image
    """
    try:
        if not GROQ_API_KEY:
            raise ValueError("Groq API key has not been set (GROQ_API_KEY).")

        client = Groq(api_key=GROQ_API_KEY)

        # Prova prima con il modello configurato per immagini
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": "Describe in detail what you see in this image. Provide an accurate and complete description of objects, people, context and any visible text."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_str}",
                                },
                            },
                        ],
                    }
                ],
                model=IMG_DESC_MODEL_LG,  # Use configured model for image description
                max_tokens=300,
                temperature=0.1,
            )
            
            caption = chat_completion.choices[0].message.content
            logger.info(f"Caption generated successfully using {IMG_DESC_MODEL_LG}")
            return caption if caption else "Image not describable"
            
        except Exception as vision_error:
            logger.warning(f"Model {IMG_DESC_MODEL_LG} does not support images: {vision_error}")
            return get_caption_alternative(base64_str)

    except Exception as e:
        logger.error(f"Error generating caption with vision model: {e}")
        return get_caption_alternative(base64_str)

def get_image_text(base64_str: str) -> str:
    """
    Extracts text from an image using OCR (Tesseract).
    
    Args:
        base64_str: Base64 string of the image
        
    Returns:
        Text extracted from the image
    """
    try:
        # Decode and convert image
        image_data = base64.b64decode(base64_str)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Unable to decode image")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return pytesseract.image_to_string(image_rgb)
        
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return ""

def get_caption_alternative(base64_str: str) -> str:
    """
    Alternative version that combines OCR + object analysis to generate a caption.
    
    Args:
        base64_str: Base64 string of the image
        
    Returns:
        Textual description of the image
    """
    try:
        # Combina informazioni da OCR e object detection
        ocr_text = get_image_text(base64_str)
        detected_objects = get_detected_objects(base64_str)
        
        caption_parts = []
        
        # Aggiungi oggetti rilevati
        if detected_objects:
            unique_objects = list(set(detected_objects))
            if len(unique_objects) == 1:
                caption_parts.append(f"Immagine contenente: {unique_objects[0]}")
            else:
                caption_parts.append(f"Immagine contenente: {', '.join(unique_objects[:3])}")
        
        # Aggiungi testo OCR se presente
        if ocr_text.strip():
            clean_text = ocr_text.strip().replace('\n', ' ')[:100]
            caption_parts.append(f"Con testo visibile: {clean_text}")
        
        # Combina le parti
        if caption_parts:
            return ". ".join(caption_parts)
        else:
            return "Immagine senza contenuto testuale riconoscibile"
            
    except Exception as e:
        logger.error(f"Errore nella caption alternativa: {e}")
        return "Immagine non analizzabile"
    
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